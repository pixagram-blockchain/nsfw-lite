import { defineConfig } from "tsup";

// One self-contained bundle for the mnv4 variant. It emits:
//   dist/mnv4/index.{js,cjs,d.ts}   — public API, embeds the model
//   dist/mnv4/worker.{js,d.ts}      — its worker copy (no model)
// The index references its worker via
//   new Worker(new URL("./worker.js", import.meta.url))
// which Vite / webpack 5 / Rollup detect statically and bundle; esbuild leaves
// that URL intact so it resolves to the sibling dist/mnv4/worker.js.
const VARIANTS = ["mnv4"] as const;

const entry: Record<string, string> = {};
for (const v of VARIANTS) {
  entry[`${v}/index`] = `src/variants/${v}/index.ts`;
  // Same source under one entry key per variant → standalone dist/<v>/worker.js.
  entry[`${v}/worker`] = "src/worker.ts";
}

export default defineConfig({
  entry,
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: true,
  target: "es2020",
  // No code-splitting: each variant's model stays embedded in its own index and
  // every worker URL resolves to a single emitted dist/<variant>/worker.js.
  splitting: false,
  // onnxruntime-web is a PEER dependency: never bundle it into our output.
  // The consumer's bundler resolves it for both the main chunk and the worker.
  external: ["onnxruntime-web", "onnxruntime-web/wasm", "onnxruntime-web/webgpu"],
});
