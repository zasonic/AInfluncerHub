import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(async () => ({
  plugins: [react()],
  clearScreen: false,
  server: {
    port:        1420,
    strictPort:  true,
    watch: {
      ignored: ["**/src-tauri/**", "**/python/**"],
    },
  },
  envPrefix: ["VITE_", "TAURI_"],
  build: {
    target:          process.env.TAURI_PLATFORM === "windows" ? "chrome105" : "safari13",
    minify:          !process.env.TAURI_DEBUG ? "esbuild" : false,
    sourcemap:       !!process.env.TAURI_DEBUG,
    rollupOptions: {
      output: {
        manualChunks: {
          react:  ["react", "react-dom"],
          vendor: ["zustand", "lucide-react"],
        },
      },
    },
  },
}));
