import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'IKMS Query Planner — AI-Powered Multi-Agent RAG',
  description:
    'Intelligent Multi-Agent RAG system with Query Planning & Decomposition powered by Google Gemini and Pinecone.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
