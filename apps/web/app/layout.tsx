// app/layout.tsx â€” layout racine, reste neutre
import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Lamoureux AI Platform",
  description: "Institutional trading cockpit with private AI research",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
