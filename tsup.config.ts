import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    // Built as its own chunk so the main bundle can reference it via
    //   new Worker(new URL("./worker.js", import.meta.url))
    // which Vite / webpack 5 / Rollup statically detect and bundle.
    worker: "src/worker.ts",
  },
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: true,
  target: "es2020",
  // Keep the worker as a standalone file (no code-splitting) so the URL
  // reference resolves to a single emitted dist/worker.js.
  splitting: false,
  // onnxruntime-web is a PEER dependency: never bundle it into our output.
  // The consumer's bundler resolves it for both the main chunk and the worker.
  external: ["onnxruntime-web", "onnxruntime-web/wasm", "onnxruntime-web/webgpu"],
});
