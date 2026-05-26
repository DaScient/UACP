import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title:       'UAP Intelligence Hub',
  description: 'Global UAP observation aggregation and analysis platform.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-uap-bg text-gray-100 min-h-screen">{children}</body>
    </html>
  );
}
