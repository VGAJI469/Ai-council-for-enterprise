/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/debate',
        destination: 'http://localhost:8000/council/debate',
      },
    ]
  }
}

export default nextConfig

