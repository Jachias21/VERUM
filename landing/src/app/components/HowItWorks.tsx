"use client";

import { useState, useEffect, useRef } from "react";
import Image from "next/image";
import { motion, AnimatePresence } from "framer-motion";
import { useLanguage } from "../context/LanguageContext";

const steps = {
  es: [
    {
      num: "01",
      mascot: "/mascot/verum-saludando.png",
      title: "Envías el contenido",
      desc: "Manda la imagen o texto sospechoso al bot de Telegram",
    },
    {
      num: "02",
      mascot: "/mascot/verum-lupa.png",
      title: "VERUM analiza",
      desc: "Aplica visión forense y análisis de frecuencias para detectar manipulación",
    },
    {
      num: "03",
      mascot: "/mascot/verum-focus.png",
      title: "Verifica las fuentes",
      desc: "Cruza los datos con bases de verificación oficiales y artículos de fact-checking",
    },
    {
      num: "04",
      mascot: "/mascot/verum-escudo.png",
      title: "Recibes el veredicto",
      desc: "Respuesta clara, explicada y con fuentes en menos de 15 segundos",
    },
  ],
  en: [
    {
      num: "01",
      mascot: "/mascot/verum-saludando.png",
      title: "You send the content",
      desc: "Send the suspicious image or text to the Telegram bot",
    },
    {
      num: "02",
      mascot: "/mascot/verum-lupa.png",
      title: "VERUM analyzes",
      desc: "Applies forensic vision and frequency analysis to detect manipulation",
    },
    {
      num: "03",
      mascot: "/mascot/verum-focus.png",
      title: "Verifies sources",
      desc: "Cross-references data with official verification databases",
    },
    {
      num: "04",
      mascot: "/mascot/verum-escudo.png",
      title: "You get the verdict",
      desc: "Clear, explained response with sources in under 15 seconds",
    },
  ],
};

const translations = {
  es: {
    eyebrow: "CÓMO FUNCIONA",
    heading: "Cuatro pasos, un veredicto",
  },
  en: {
    eyebrow: "HOW IT WORKS",
    heading: "Four steps, one verdict",
  },
};

export default function HowItWorks() {
  const { lang } = useLanguage();
  const t = translations[lang];
  const stepsData = steps[lang];
  const [activeStep, setActiveStep] = useState(0);
  const stepRefs = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    const observers: IntersectionObserver[] = [];

    stepRefs.current.forEach((ref, index) => {
      if (!ref) return;
      const observer = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) {
            setActiveStep(index);
          }
        },
        { threshold: 0.6 }
      );
      observer.observe(ref);
      observers.push(observer);
    });

    return () => observers.forEach((obs) => obs.disconnect());
  }, [lang]);

  return (
    <section id="how" className="bg-white py-24 md:py-32">
      <div className="mx-auto max-w-7xl px-6">
        {/* Header */}
        <div className="mb-16 text-center md:mb-24">
          <motion.span
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-4 inline-block rounded-full border border-teal bg-teal-light/40 px-4 py-1 font-[var(--font-poppins)] text-xs font-semibold uppercase tracking-widest text-teal"
          >
            {t.eyebrow}
          </motion.span>

          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="mt-4 font-[var(--font-poppins)] text-3xl font-bold text-navy md:text-4xl"
          >
            {t.heading}
          </motion.h2>
        </div>

        {/* Desktop: two columns with sticky mascot */}
        <div className="hidden md:grid md:grid-cols-2 md:gap-16">
          {/* Left: Steps */}
          <div>
            {stepsData.map((step, index) => (
              <div
                key={step.num}
                ref={(el) => { stepRefs.current[index] = el; }}
                className="relative py-20"
              >
                {/* Decorative number */}
                <span className="absolute -top-2 left-0 font-[var(--font-poppins)] text-8xl font-bold text-teal-light/60 select-none">
                  {step.num}
                </span>
                <div className="relative">
                  <h3 className="mb-3 font-[var(--font-poppins)] text-3xl font-bold text-navy">
                    {step.title}
                  </h3>
                  <p className="max-w-md text-lg leading-relaxed text-text-secondary">
                    {step.desc}
                  </p>
                </div>
              </div>
            ))}
          </div>

          {/* Right: Sticky mascot */}
          <div className="flex items-start justify-center">
            <div className="sticky top-[120px] flex items-center justify-center">
              <AnimatePresence mode="wait">
                <motion.div
                  key={stepsData[activeStep].mascot}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ duration: 0.3 }}
                >
                  <Image
                    src={stepsData[activeStep].mascot}
                    alt={`VERUM - ${stepsData[activeStep].title}`}
                    width={380}
                    height={380}
                    className="object-contain"
                  />
                </motion.div>
              </AnimatePresence>
            </div>
          </div>
        </div>

        {/* Mobile: single column with inline mascot */}
        <div className="space-y-16 md:hidden">
          {stepsData.map((step) => (
            <motion.div
              key={step.num}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="text-center"
            >
              <Image
                src={step.mascot}
                alt={`VERUM - ${step.title}`}
                width={200}
                height={200}
                className="mx-auto mb-6 object-contain"
              />
              <span className="font-[var(--font-poppins)] text-5xl font-bold text-teal-light/60">
                {step.num}
              </span>
              <h3 className="mt-2 font-[var(--font-poppins)] text-2xl font-bold text-navy">
                {step.title}
              </h3>
              <p className="mt-3 text-base leading-relaxed text-text-secondary">
                {step.desc}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
