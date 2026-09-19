import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Aug 30 2026 React rebuild: relative `base` so the built index.html's
// asset URLs work whether the FastAPI service ends up serving them from
// "/" (the normal case, see main.py's static mount) or from some other
// sub-path in the future - no hardcoded absolute origin either way.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
