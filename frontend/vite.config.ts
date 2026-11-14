/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: /api is proxied to the Flask backend (python -m backend). Build: frontend/dist, which the backend serves.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:5000" } },
  // the 3D hero chunk (three.js, about 1 MB) is lazy and loads only with the Overview hero
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 1100 },
  test: {
    environment: "jsdom", globals: true, setupFiles: ["src/test/setup.ts"], css: false,
    unstubGlobals: true, restoreMocks: true, testTimeout: 20000,
  },
});
