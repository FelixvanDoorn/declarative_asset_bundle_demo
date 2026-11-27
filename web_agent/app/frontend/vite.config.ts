import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  build: {
    // FastAPI (server.py) serves this folder at /.
    outDir: "../static",
    emptyOutDir: true,
  },
  server: {
    // `npm run dev`: forward API calls to the local FastAPI server (python server.py).
    proxy: { "/api": "http://localhost:8000" },
  },
});
