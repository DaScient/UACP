/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static-exportable for GitHub Pages.
  output: 'export',
  images: { unoptimized: true },
  trailingSlash: true,
};
module.exports = nextConfig;
