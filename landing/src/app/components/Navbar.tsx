"use client";

import { useState, useEffect } from "react";
import Image from "next/image";
import { useLanguage } from "../context/LanguageContext";

const translations = {
  es: {
    howItWorks: "Cómo funciona",
    whyVerum: "Por qué VERUM",
    cta: "Abrir en Telegram",
  },
  en: {
    howItWorks: "How it works",
    whyVerum: "Why VERUM",
    cta: "Open in Telegram",
  },
};

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const { lang, setLang } = useLanguage();
  const t = translations[lang];

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <nav
      id="navbar"
      className={`fixed top-0 left-0 right-0 z-50 transition-shadow duration-300 ${
        scrolled ? "shadow-md bg-white/95 backdrop-blur-sm" : "bg-white"
      }`}
    >
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
        {/* Left: Logo + Name */}
        <a href="#" className="flex items-center gap-2">
          <Image
            src="/logo.png"
            alt="VERUM logo"
            width={36}
            height={36}
            className="h-9 w-9 object-contain"
          />
          <span className="font-[var(--font-poppins)] text-xl font-bold text-navy">
            VERUM
          </span>
        </a>

        {/* Right: Nav links + Lang selector + CTA */}
        <div className="flex items-center gap-6">
          {/* Navigation links — hidden on mobile */}
          <a
            href="#how"
            className="hidden text-sm font-medium text-text-secondary transition-colors hover:text-navy md:block"
          >
            {t.howItWorks}
          </a>
          <a
            href="#why"
            className="hidden text-sm font-medium text-text-secondary transition-colors hover:text-navy md:block"
          >
            {t.whyVerum}
          </a>

          {/* Language selector — hidden on mobile */}
          <div className="hidden items-center gap-1 md:flex">
            <button
              onClick={() => setLang("es")}
              className={`rounded-md px-2 py-1 font-[var(--font-poppins)] text-xs font-semibold transition-colors ${
                lang === "es"
                  ? "bg-teal-light text-teal"
                  : "text-text-secondary hover:text-navy"
              }`}
            >
              ES
            </button>
            <button
              onClick={() => setLang("en")}
              className={`rounded-md px-2 py-1 font-[var(--font-poppins)] text-xs font-semibold transition-colors ${
                lang === "en"
                  ? "bg-teal-light text-teal"
                  : "text-text-secondary hover:text-navy"
              }`}
            >
              EN
            </button>
          </div>

          {/* CTA Button */}
          <a
            href="https://t.me/Verum_tfm_bot"
            target="_blank"
            rel="noopener noreferrer"
            id="navbar-cta"
            className="rounded-full bg-teal px-5 py-2 font-[var(--font-poppins)] text-sm font-semibold text-navy transition-colors hover:bg-teal-hover"
          >
            {t.cta}
          </a>
        </div>
      </div>
    </nav>
  );
}
