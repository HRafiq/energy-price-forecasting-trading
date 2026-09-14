import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The Python API serves dist/ at "/" and the JSON endpoints under /api, so the
// app only ever calls relative /api URLs. In development Vite forwards them.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/",
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "dist",
    rolldownOptions: {
      output: {
        // Vite 8 bundles with Rolldown; codeSplitting groups replace Rollup's manualChunks.
        codeSplitting: {
          groups: [
            { name: "react", test: /node_modules[\\/](react|react-dom|scheduler)[\\/]/, priority: 20 },
            // recharts with its d3, redux and helper dependencies
            { name: "recharts", test: /node_modules[\\/]/, priority: 10 },
          ],
        },
      },
    },
  },
});
