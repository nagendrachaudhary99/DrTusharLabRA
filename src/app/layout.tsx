import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Nav } from "@/components/nav";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "iTRIALSPACE — Agentic Virtual Imaging Trials",
  description:
    "Programmable virtual imaging trials for lung-nodule AI: synthetic patients, simulated CT scanners, LangGraph multi-agent orchestration and RAG-grounded reporting.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col">
        <Nav />
        <main className="flex-1">{children}</main>
        <footer className="border-t border-line py-6 text-center text-xs text-ink-3">
          iTRIALSPACE prototype · Next.js · LangChain.js · LangGraph.js · RAG · Synthetic data only. Not for clinical use.
        </footer>
      </body>
    </html>
  );
}
