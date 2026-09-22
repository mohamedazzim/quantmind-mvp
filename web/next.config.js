/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    // NOTE: In Docker, /api/v1/* requests are handled by the App Router
    // catch-all at src/app/api/v1/[...path]/route.ts which reads
    // INTERNAL_API_URL at request time. This rewrite is only used in
    // local development (outside Docker) where 127.0.0.1:8000 is reachable.
    return [
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:8000/api/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
