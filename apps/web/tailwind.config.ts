import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#090c11",
        steel: "#111823",
        panel: "#161f2c",
        line: "#243347",
        accent: "#7dd3fc",
        gain: "#26d07c",
        loss: "#ff6b6b",
        sand: "#f0d7a1"
      },
      fontFamily: {
        sans: ["'IBM Plex Sans'", "sans-serif"],
        mono: ["'IBM Plex Mono'", "monospace"]
      },
      boxShadow: {
        bloom: "0 24px 80px rgba(10, 18, 32, 0.45)"
      },
      backgroundImage: {
        grid: "linear-gradient(rgba(125, 211, 252, 0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(125, 211, 252, 0.08) 1px, transparent 1px)"
      }
    }
  },
  plugins: []
};

export default config;

