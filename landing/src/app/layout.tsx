import type { Metadata } from "next";
import { Poppins, Nunito } from "next/font/google";
import "./globals.css";
import { LanguageProvider } from "./context/LanguageContext";

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["600", "700"],
  variable: "--font-poppins",
  display: "swap",
});

const nunito = Nunito({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-nunito",
  display: "swap",
});

export const metadata: Metadata = {
  title: "VERUM — Asistente de Verdad",
  description: "Tu perito forense de bolsillo contra las Fake News",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es" className={`${poppins.variable} ${nunito.variable}`}>
      <body>
        <LanguageProvider>
          <main>{children}</main>
        </LanguageProvider>
      </body>
    </html>
  );
}
