import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// export default defineConfig({
//   plugins: [react()],
// })
export default defineConfig({
  plugins: [react()],
  server: {
      proxy: {
          '/query': {
              target: 'http://localhost:8001',
              changeOrigin: true,
          },
      },
  },
});