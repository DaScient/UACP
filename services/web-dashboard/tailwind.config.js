/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx}',
    './pages/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        'uap-bg':   '#0b0f17',
        'uap-card': '#111827',
        'uap-accent': '#22d3ee',
      },
    },
  },
  plugins: [],
};
