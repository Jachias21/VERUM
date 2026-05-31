"use client";

import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import WhatIsVerum from "./components/WhatIsVerum";
import HowItWorks from "./components/HowItWorks";
import WhyVerum from "./components/WhyVerum";
import CTAFinal from "./components/CTAFinal";

export default function Home() {
  return (
    <>
      <Navbar />
      <Hero />
      <WhatIsVerum />
      <HowItWorks />
      <WhyVerum />
      <CTAFinal />
    </>
  );
}
