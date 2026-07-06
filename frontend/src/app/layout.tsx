import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Crypto Bot — OKX Trading AI",
  description: "Hibrit TA+ML kripto trading botu",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="tr" className="h-full">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
