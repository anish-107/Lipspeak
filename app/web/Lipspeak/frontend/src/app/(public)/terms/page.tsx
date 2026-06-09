/** page.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Terms and conditions page for LipSpeak AI.
 * @date: 10 May 2026
 * @returns: Terms page component.
 * 
 */


// Imports
import { Navbar } from "@/components/common/Navbar";
import { Footer } from "@/components/common/Footer";
import { BackgroundBeams } from "@/components/ui/aceternity/background-beams";


// TermsPage Component
export default function TermsPage() {
  // Render
  return (
    <main
      className="
        min-h-screen
        bg-background
        text-foreground
      "
    >
      <Navbar />

      <section
        className="
          relative
          overflow-hidden
          px-6
          py-24
          md:py-32
        "
      >
        <BackgroundBeams />
        {/* Background Glow */}
        <div
          className="
            absolute
            inset-0
            pointer-events-none
            bg-[radial-gradient(circle_at_top,rgba(99,102,241,0.08),transparent_45%)]
            dark:bg-[radial-gradient(circle_at_top,rgba(99,102,241,0.15),transparent_45%)]
          "
        />

        <div className="relative mx-auto max-w-5xl">
          {/* Header */}
          <div className="mb-16 text-center">
            <h1
              className="
                text-4xl
                font-black
                tracking-tight
                md:text-6xl
              "
            >
              Terms &
              <span className="gradient-text block">
                Conditions
              </span>
            </h1>

            <p
              className="
                mx-auto
                mt-6
                max-w-2xl
                text-lg
                text-muted-foreground
              "
            >
              Please read these terms carefully before accessing
              or using LipSpeak AI services.
            </p>

            <p
              className="
                mt-4
                text-sm
                text-muted-foreground
              "
            >
              Last Updated: June 2026
            </p>
          </div>

          {/* Content */}
          <div
            className="
              glass-card
              ai-border
              rounded-4xl
              p-8
              md:p-12
            "
          >
            <div className="space-y-12">
              {/* Section */}
              <div>
                <h2 className="text-2xl font-bold">
                  1. Acceptance of Terms
                </h2>

                <p
                  className="
                    mt-4
                    leading-8
                    text-muted-foreground
                  "
                >
                  By accessing or using LipSpeak AI, you agree to
                  comply with and be bound by these Terms and
                  Conditions. If you do not agree, please discontinue
                  use of the platform immediately.
                </p>
              </div>

              {/* Section */}
              <div>
                <h2 className="text-2xl font-bold">
                  2. Platform Usage
                </h2>

                <p
                  className="
                    mt-4
                    leading-8
                    text-muted-foreground
                  "
                >
                  Users must use the platform responsibly and only for
                  lawful purposes. Uploading malicious, harmful,
                  unauthorized, or illegal content is strictly
                  prohibited.
                </p>
              </div>

              {/* Section */}
              <div>
                <h2 className="text-2xl font-bold">
                  3. AI Predictions & Accuracy
                </h2>

                <p
                  className="
                    mt-4
                    leading-8
                    text-muted-foreground
                  "
                >
                  LipSpeak AI utilizes machine learning and computer
                  vision technologies to generate predictions. While
                  every effort is made to maximize accuracy, outputs
                  may contain errors and should not be considered
                  legally, medically, or professionally definitive.
                </p>
              </div>

              {/* Section */}
              <div>
                <h2 className="text-2xl font-bold">
                  4. User Content
                </h2>

                <p
                  className="
                    mt-4
                    leading-8
                    text-muted-foreground
                  "
                >
                  Users retain ownership of uploaded content. By using
                  the platform, you grant LipSpeak AI permission to
                  process uploaded media solely for generating
                  predictions and providing requested services.
                </p>
              </div>

              {/* Section */}
              <div>
                <h2 className="text-2xl font-bold">
                  5. Account Responsibility
                </h2>

                <p
                  className="
                    mt-4
                    leading-8
                    text-muted-foreground
                  "
                >
                  Users are responsible for maintaining the security
                  of their credentials and all activity occurring under
                  their accounts.
                </p>
              </div>

              {/* Section */}
              <div>
                <h2 className="text-2xl font-bold">
                  6. Service Availability
                </h2>

                <p
                  className="
                    mt-4
                    leading-8
                    text-muted-foreground
                  "
                >
                  Services may be modified, suspended, or temporarily
                  unavailable due to maintenance, infrastructure
                  upgrades, or unforeseen technical issues.
                </p>
              </div>

              {/* Section */}
              <div>
                <h2 className="text-2xl font-bold">
                  7. Limitation of Liability
                </h2>

                <p
                  className="
                    mt-4
                    leading-8
                    text-muted-foreground
                  "
                >
                  LipSpeak AI and its contributors shall not be liable
                  for any indirect, incidental, or consequential
                  damages resulting from use of the platform or
                  reliance on generated predictions.
                </p>
              </div>

              {/* Section */}
              <div>
                <h2 className="text-2xl font-bold">
                  8. Changes to Terms
                </h2>

                <p
                  className="
                    mt-4
                    leading-8
                    text-muted-foreground
                  "
                >
                  We reserve the right to update these Terms and
                  Conditions at any time. Continued use of the platform
                  after updates constitutes acceptance of the revised
                  terms.
                </p>
              </div>

              {/* Section */}
              <div>
                <h2 className="text-2xl font-bold">
                  9. Contact Information
                </h2>

                <p
                  className="
                    mt-4
                    leading-8
                    text-muted-foreground
                  "
                >
                  For questions regarding these Terms and Conditions,
                  please contact the LipSpeak AI team through the
                  official support channels.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}

