/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        neon: {
          green: "#39ff14",
          blue: "#04d9ff",
          pink: "#ff6ec7",
          yellow: "#fff01f",
          red: "#ff073a",
        },
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        glow: "glow 2s ease-in-out infinite alternate",
        scan: "scan 3s linear infinite",
      },
      keyframes: {
        glow: {
          "0%": { boxShadow: "0 0 5px #04d9ff, 0 0 10px #04d9ff" },
          "100%": { boxShadow: "0 0 20px #04d9ff, 0 0 40px #04d9ff" },
        },
        scan: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100%)" },
        },
      },
    },
  },
  plugins: [require("@tailwindcss/forms")],
};
