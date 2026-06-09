/** page.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Privacy policy page for LipSpeak AI platform.
 * @date: 10 May 2026
 * @returns: Privacy policy page component.
 * 
 */


// Imports
import { Navbar } from "@/components/common/Navbar";
import { Footer } from "@/components/common/Footer";
import { BackgroundBeams } from "@/components/ui/aceternity/background-beams";


// PrivacyPage Component
export default function PrivacyPage() {
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
            bg-[radial-gradient(circle_at_top,rgba(34,211,238,0.08),transparent_45%)]
            dark:bg-[radial-gradient(circle_at_top,rgba(34,211,238,0.15),transparent_45%)]
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
              Privacy
              <span className="gradient-text block">
                Policy
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
              Your privacy, security, and control over your data are
              important to us. This policy explains how LipSpeak AI
              collects, uses, and protects your information.
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
                  1. Information We Collect
                </h2>

                <p
                  className="
                    mt-4
                    leading-8
                    text-muted-foreground
                  "
                >
                  LipSpeak AI may collect account information,
                  authentication details, uploaded videos, and
                  platform usage analytics necessary to provide and
                  improve our services.
                </p>
              </div>

              {/* Section */}
              <div>
                <h2 className="text-2xl font-bold">
                  2. How We Use Your Information
                </h2>

                <p
                  className="
                    mt-4
                    leading-8
                    text-muted-foreground
                  "
                >
                  Information is used to authenticate users,
                  process uploaded media, generate AI predictions,
                  improve model performance, and enhance the overall
                  user experience.
                </p>
              </div>

              {/* Section */}
              <div>
                <h2 className="text-2xl font-bold">
                  3. AI Processing
                </h2>

                <p
                  className="
                    mt-4
                    leading-8
                    text-muted-foreground
                  "
                >
                  Uploaded video content may be analyzed by machine
                  learning models solely for visual speech recognition,
                  transcript generation, accessibility support,
                  and platform functionality.
                </p>
              </div>

              {/* Section */}
              <div>
                <h2 className="text-2xl font-bold">
                  4. Data Security
                </h2>

                <p
                  className="
                    mt-4
                    leading-8
                    text-muted-foreground
                  "
                >
                  We implement industry-standard security measures,
                  encryption practices, secure APIs, and protected
                  cloud infrastructure to safeguard user data against
                  unauthorized access.
                </p>
              </div>

              {/* Section */}
              <div>
                <h2 className="text-2xl font-bold">
                  5. Data Retention
                </h2>

                <p
                  className="
                    mt-4
                    leading-8
                    text-muted-foreground
                  "
                >
                  Data is retained only as long as necessary to provide
                  services, maintain platform integrity, comply with
                  legal obligations, or improve AI system performance.
                </p>
              </div>

              {/* Section */}
              <div>
                <h2 className="text-2xl font-bold">
                  6. User Rights
                </h2>

                <p
                  className="
                    mt-4
                    leading-8
                    text-muted-foreground
                  "
                >
                  Users may request access to, correction of,
                  or deletion of personal information where
                  applicable. Account removal requests may also
                  be submitted through platform support channels.
                </p>
              </div>

              {/* Section */}
              <div>
                <h2 className="text-2xl font-bold">
                  7. Third-Party Services
                </h2>

                <p
                  className="
                    mt-4
                    leading-8
                    text-muted-foreground
                  "
                >
                  LipSpeak AI may utilize trusted third-party
                  infrastructure providers, authentication services,
                  and analytics platforms necessary for operation
                  and performance monitoring.
                </p>
              </div>

              {/* Section */}
              <div>
                <h2 className="text-2xl font-bold">
                  8. Changes to This Policy
                </h2>

                <p
                  className="
                    mt-4
                    leading-8
                    text-muted-foreground
                  "
                >
                  This Privacy Policy may be updated periodically.
                  Continued use of the platform following updates
                  constitutes acceptance of the revised policy.
                </p>
              </div>

              {/* Section */}
              <div>
                <h2 className="text-2xl font-bold">
                  9. Contact Us
                </h2>

                <p
                  className="
                    mt-4
                    leading-8
                    text-muted-foreground
                  "
                >
                  If you have questions regarding this Privacy Policy
                  or your personal data, please contact the LipSpeak AI
                  team through the official support channels.
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
