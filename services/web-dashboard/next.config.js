/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static-exportable for GitHub Pages.
  output: 'export',
  images: { unoptimized: true },
  trailingSlash: true,
  // GitHub Pages project sites are served under /<repo-name>/
  basePath: '/UACP',
  assetPrefix: '/UACP/',
};
module.exports = nextConfig;
