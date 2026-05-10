import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Hedge Fund Platform",
  description: "Institutional trading cockpit"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

