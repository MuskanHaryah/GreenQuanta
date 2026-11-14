/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: /api is proxied to the Flask backend (python -m backend). Build: frontend/dist, which the backend serves.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:5000" } },
  build: { outDir: "dist", emptyOutDir: true },
  test: {
    environment: "jsdom", globals: true, setupFiles: ["src/test/setup.ts"], css: false,
    unstubGlobals: true, restoreMocks: true,
  },
});
