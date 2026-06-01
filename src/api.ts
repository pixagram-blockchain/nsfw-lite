/**
 * api.ts — variant-agnostic detector engine. All the heavy machinery
 * (image decoding, batching, worker vs main-thread implementations) lives at
 * module scope and is shared across every model variant. The only
 * variant-specific input is the {@link Embedded} model, injected via
 * {@link makeNsfwApi}; each `src/variants/<v>/index.ts` calls it once and
 * re-exports the result as that variant's public API.
 */
import * as core from "./core.js";
import type { Embedded } from "./assets.js";
import type {
  DetectorOptions,
  ImageSource,
  NsfwResult,
  PreprocessConfig,
  Thresholds,
} from "./types.js";

const DEFAULT_THRESHOLDS: Thresholds = { nsfw: 0.5 };

// --- Public surface (explicit interfaces) ----------------------------------
// Hand-written rather than `InstanceType<typeof NsfwDetector>`: the class has a
// private constructor, so the InstanceType form fails to satisfy a structural
// class type. These interfaces keep the emitted .d.ts clean per variant.

export interface NsfwDetectorInstance {
  /** Whether inference runs in a Worker or on the main thread. */
  readonly backend: "worker" | "direct";
  /** Classify one image. Accepts ImageData, ImageBitmap, <img>/<canvas>, Blob, or a URL. */
  classify(source: ImageSource): Promise<NsfwResult>;
  /** Release the worker / ORT session. */
  dispose(): void;
}

export interface NsfwDetectorClass {
  create(opts?: DetectorOptions): Promise<NsfwDetectorInstance>;
}

export interface NsfwApi {
  NsfwDetector: NsfwDetectorClass;
  classify(source: ImageSource, opts?: DetectorOptions): Promise<NsfwResult>;
  warmup(opts?: DetectorOptions): Promise<void>;
  disposeShared(): void;
}

// --- Image decoding (main thread only) -------------------------------------

function makeCanvas(w: number, h: number): OffscreenCanvas | HTMLCanvasElement {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(w, h);
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

// One reused canvas for decoding sources to pixels, instead of allocating a new
// one per classify(). Resized on demand (resizing clears it). toPixelData has no
// `await` between the draw and getImageData, so even overlapping classify()
// calls can't interleave inside that critical section — sharing is safe.
type AnyCanvas = OffscreenCanvas | HTMLCanvasElement;
type Any2DContext = OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;
let decodeCanvas: AnyCanvas | null = null;
let decodeCtx: Any2DContext | null = null;

function decodeContext(w: number, h: number): Any2DContext {
  if (!decodeCanvas) {
    decodeCanvas = makeCanvas(w, h);
    decodeCtx = (decodeCanvas as OffscreenCanvas).getContext("2d", {
      willReadFrequently: true,
    }) as Any2DContext | null;
  } else if (decodeCanvas.width !== w || decodeCanvas.height !== h) {
    decodeCanvas.width = w; // clears the canvas
    decodeCanvas.height = h;
  }
  if (!decodeCtx) throw new Error("@pixagram/nsfw-lite: 2D canvas context unavailable");
  return decodeCtx;
}

function isImageData(x: unknown): x is ImageData {
  return typeof ImageData !== "undefined" && x instanceof ImageData;
}

async function toPixelData(source: ImageSource): Promise<core.PixelData> {
  if (isImageData(source)) {
    // Copy: the worker path transfers (detaches) the buffer, and we must not
    // detach a caller-owned ImageData.
    return {
      data: new Uint8ClampedArray(source.data),
      width: source.width,
      height: source.height,
    };
  }

  let bitmap: ImageBitmap | null = null;
  let drawable: CanvasImageSource;

  if (typeof source === "string") {
    const res = await fetch(source);
    bitmap = await createImageBitmap(await res.blob());
    drawable = bitmap;
  } else if (typeof Blob !== "undefined" && source instanceof Blob) {
    bitmap = await createImageBitmap(source);
    drawable = bitmap;
  } else {
    drawable = source as CanvasImageSource;
  }

  const anyDrawable = drawable as unknown as {
    width?: number;
    height?: number;
    videoWidth?: number;
    videoHeight?: number;
  };
  const w = anyDrawable.width ?? anyDrawable.videoWidth ?? 0;
  const h = anyDrawable.height ?? anyDrawable.videoHeight ?? 0;
  if (!w || !h) throw new Error("@pixagram/nsfw-lite: could not determine image dimensions");

  const ctx = decodeContext(w, h);
  ctx.clearRect(0, 0, w, h); // reused canvas: clear so transparent sources don't bleed
  ctx.drawImage(drawable, 0, 0);
  const id = ctx.getImageData(0, 0, w, h);
  if (bitmap) bitmap.close();
  return { data: id.data, width: w, height: h };
}

// --- Implementations --------------------------------------------------------

interface Impl {
  classify(source: ImageSource): Promise<NsfwResult>;
  dispose(): void;
}

const DEFAULT_MAX_BATCH = 8;
const DEFAULT_BATCH_DELAY_MS = 12;

type BatchRunner = (items: core.PixelData[]) => Promise<NsfwResult[]>;

/**
 * Coalesces individual classify() calls into batched inferences. submit()
 * returns a promise for that single image, but requests arriving within delayMs
 * (or until maxBatch is reached) are run together through runBatch — so the
 * fixed per-call cost (GPU upload / kernel launch / readback) is amortised
 * across the whole batch. Overflow beyond maxBatch rolls into the next batch.
 */
class Batcher {
  private queue: Array<{
    px: core.PixelData;
    resolve: (r: NsfwResult) => void;
    reject: (e: Error) => void;
  }> = [];
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private run: BatchRunner,
    private maxBatch: number,
    private delayMs: number
  ) {}

  submit(px: core.PixelData): Promise<NsfwResult> {
    return new Promise<NsfwResult>((resolve, reject) => {
      this.queue.push({ px, resolve, reject });
      if (this.queue.length >= this.maxBatch) {
        this.flush(); // batch full → run immediately, don't wait for the timer
      } else if (this.timer === null) {
        this.timer = setTimeout(() => this.flush(), this.delayMs);
      }
    });
  }

  private flush(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    if (this.queue.length === 0) return;
    const batch = this.queue.splice(0, this.maxBatch);
    this.run(batch.map((b) => b.px))
      .then((results) => {
        for (let i = 0; i < batch.length; i++) {
          const r = results[i];
          if (r) batch[i]!.resolve(r);
          else batch[i]!.reject(new Error("@pixagram/nsfw-lite: missing batch result"));
        }
      })
      .catch((err) => {
        const e = err instanceof Error ? err : new Error(String(err));
        for (const b of batch) b.reject(e);
      });
    // More than one batch's worth was queued — schedule the remainder.
    if (this.queue.length > 0 && this.timer === null) {
      this.timer = setTimeout(() => this.flush(), this.delayMs);
    }
  }
}

class DirectImpl implements Impl {
  private batcher: Batcher;

  private constructor(
    private sess: core.Session,
    cfg: PreprocessConfig,
    labels: string[],
    thresholds: Thresholds,
    maxBatch: number,
    delayMs: number
  ) {
    this.batcher = new Batcher(
      (items) => core.classifyBatch(this.sess, items, cfg, labels, thresholds),
      maxBatch,
      delayMs
    );
  }

  static async create(
    bytes: Uint8Array,
    cfg: PreprocessConfig,
    labels: string[],
    thresholds: Thresholds,
    opts: DetectorOptions
  ): Promise<DirectImpl> {
    const sess = await core.loadSession(bytes, {
      wasmPaths: opts.wasmPaths,
      numThreads: opts.numThreads,
      backend: opts.backend,
    });
    return new DirectImpl(
      sess,
      cfg,
      labels,
      thresholds,
      opts.maxBatch ?? DEFAULT_MAX_BATCH,
      opts.batchDelayMs ?? DEFAULT_BATCH_DELAY_MS
    );
  }

  async classify(source: ImageSource): Promise<NsfwResult> {
    const px = await toPixelData(source);
    return this.batcher.submit(px);
  }

  dispose(): void {
    try {
      (this.sess.session as unknown as { release?: () => void }).release?.();
    } catch {
      /* ignore */
    }
  }
}

interface WorkerResponse {
  reqId: number;
  ok: boolean;
  error?: string;
  result?: NsfwResult;
  results?: NsfwResult[];
  backend?: string;
}
type Pending = { resolve: (v: WorkerResponse) => void; reject: (e: Error) => void };

class WorkerImpl implements Impl {
  private seq = 0;
  private pending = new Map<number, Pending>();
  private batcher: Batcher;

  private constructor(private worker: Worker, maxBatch: number, delayMs: number) {
    this.worker.onmessage = (e: MessageEvent) => {
      const data = (e.data || {}) as WorkerResponse;
      const p = this.pending.get(data.reqId);
      if (!p) return;
      this.pending.delete(data.reqId);
      if (data.ok) p.resolve(data);
      else p.reject(new Error(data.error || "worker error"));
    };
    this.worker.onerror = (e: ErrorEvent) => {
      const err = new Error(e.message || "worker crashed");
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();
    };
    this.batcher = new Batcher((items) => this.runBatch(items), maxBatch, delayMs);
  }

  static async create(
    bytes: Uint8Array,
    cfg: PreprocessConfig,
    labels: string[],
    thresholds: Thresholds,
    opts: DetectorOptions
  ): Promise<WorkerImpl> {
    // Statically detectable by Vite / webpack 5 / Rollup so the worker chunk
    // is bundled and its URL rewritten. Each variant's bundle emits its own
    // sibling dist/<variant>/worker.js, which this resolves against. Throws in
    // CJS / no-Worker contexts, which NsfwDetector.create() catches to fall
    // back to main-thread.
    const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    const impl = new WorkerImpl(
      worker,
      opts.maxBatch ?? DEFAULT_MAX_BATCH,
      opts.batchDelayMs ?? DEFAULT_BATCH_DELAY_MS
    );
    // Transfer the model buffer (zero-copy). The main thread no longer needs it.
    const buf = bytes.buffer.slice(
      bytes.byteOffset,
      bytes.byteOffset + bytes.byteLength
    ) as ArrayBuffer;
    await impl.rpc(
      "init",
      {
        modelBuffer: buf,
        cfg,
        labels,
        thresholds,
        opts: { wasmPaths: opts.wasmPaths, numThreads: opts.numThreads, backend: opts.backend },
      },
      [buf]
    );
    return impl;
  }

  private rpc(
    type: string,
    data: Record<string, unknown>,
    transfer: Transferable[] = []
  ): Promise<WorkerResponse> {
    return new Promise<WorkerResponse>((resolve, reject) => {
      const reqId = ++this.seq;
      this.pending.set(reqId, { resolve, reject });
      this.worker.postMessage({ type, reqId, ...data }, transfer);
    });
  }

  private async runBatch(items: core.PixelData[]): Promise<NsfwResult[]> {
    // Transfer every pixel buffer (zero-copy); each was freshly decoded.
    const transfer = items.map((p) => p.data.buffer as ArrayBuffer);
    const resp = await this.rpc("classifyBatch", { payloads: items }, transfer);
    if (!resp.results) throw new Error("@pixagram/nsfw-lite: worker returned no batch results");
    return resp.results;
  }

  async classify(source: ImageSource): Promise<NsfwResult> {
    const px = await toPixelData(source);
    return this.batcher.submit(px);
  }

  dispose(): void {
    this.worker.terminate();
    this.pending.clear();
  }
}

function canUseWorker(): boolean {
  try {
    return typeof Worker !== "undefined";
  } catch {
    return false;
  }
}

// --- Variant factory --------------------------------------------------------

/**
 * Build the public API for one model variant from its resolved
 * {@link Embedded} assets. The returned {@link NsfwApi} carries an
 * `NsfwDetector` class plus the `classify` / `warmup` / `disposeShared`
 * one-shot helpers, all closed over the supplied embedded model.
 */
export function makeNsfwApi(embedded: Embedded): NsfwApi {
  class NsfwDetector implements NsfwDetectorInstance {
    private constructor(
      private impl: Impl,
      public readonly backend: "worker" | "direct"
    ) {}

    static async create(opts: DetectorOptions = {}): Promise<NsfwDetector> {
      const cfg: PreprocessConfig = { ...embedded.preprocess, ...(opts.preprocess || {}) };
      const labels = opts.labels ?? embedded.labels;
      const thresholds: Thresholds = { ...DEFAULT_THRESHOLDS, ...(opts.thresholds || {}) };
      const bytes = opts.modelBytes ?? embedded.getModelBytes();

      const mode = opts.useWorker ?? "auto";
      const wantWorker = mode === true || (mode === "auto" && canUseWorker());

      if (wantWorker) {
        try {
          const impl = await WorkerImpl.create(bytes, cfg, labels, thresholds, opts);
          return new NsfwDetector(impl, "worker");
        } catch {
          if (opts.useWorker === true) {
            throw new Error(
              "@pixagram/nsfw-lite: worker requested but unavailable in this environment"
            );
          }
          // auto: fall through to main-thread
        }
      }

      const impl = await DirectImpl.create(bytes, cfg, labels, thresholds, opts);
      return new NsfwDetector(impl, "direct");
    }

    classify(source: ImageSource): Promise<NsfwResult> {
      return this.impl.classify(source);
    }

    dispose(): void {
      this.impl.dispose();
    }
  }

  // --- one-shot shared detector (per variant) ------------------------------
  let singleton: Promise<NsfwDetector> | null = null;

  function shared(opts?: DetectorOptions): Promise<NsfwDetector> {
    if (!singleton) singleton = NsfwDetector.create(opts);
    return singleton;
  }

  async function warmup(opts?: DetectorOptions): Promise<void> {
    await shared(opts);
  }

  async function classify(source: ImageSource, opts?: DetectorOptions): Promise<NsfwResult> {
    const detector = await shared(opts);
    return detector.classify(source);
  }

  function disposeShared(): void {
    if (!singleton) return;
    const pending = singleton;
    singleton = null;
    void pending.then((d) => d.dispose()).catch(() => undefined);
  }

  return { NsfwDetector, classify, warmup, disposeShared };
}
