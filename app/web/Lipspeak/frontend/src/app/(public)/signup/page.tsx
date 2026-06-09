/** page.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Public signup page for new user registration.
 * @date: 10 June 2026
 * @returns: Signup page component.
 *
 */


// Imports
import { Navbar } from "@/components/common/Navbar";
import { Footer } from "@/components/common/Footer";
import { SignupForm } from "@/components/auth/SignupForm";
import { BackgroundBeams } from "@/components/ui/aceternity/background-beams";
import {
  Brain,
  Sparkles,
  Users,
} from "lucide-react";


// Signup Page Component
export default function SignupPage() {
  // Render
  return (
    <main
      className="
        min-h-screen
        overflow-hidden
        bg-background
        text-foreground
      "
    >
      <Navbar />

      <section
        className="
          relative
          overflow-hidden
          px-4
          py-12
          sm:px-6
          lg:px-8
        "
      >
        <BackgroundBeams />

        <div
          className="
            relative
            z-10
            mx-auto
            max-w-7xl
          "
        >
          <div
            className="
              grid
              min-h-[75vh]
              items-center
              gap-16
              lg:grid-cols-2
            "
          >
            {/* Left Content */}
            <div className="hidden lg:block">
              <div
                className="
                  inline-flex
                  items-center
                  gap-2
                  rounded-full
                  border
                  border-primary/20
                  bg-primary/10
                  px-4
                  py-2
                  text-sm
                  font-medium
                  text-primary
                "
              >
                <Sparkles className="h-4 w-4" />
                Join The Future Of Communication
              </div>

              <h1
                className="
                  mt-8
                  text-5xl
                  font-black
                  leading-tight
                  tracking-tight
                  xl:text-6xl
                "
              >
                Create Your
                <span
                  className="
                    block
                    gradient-text
                  "
                >
                  LipSpeak AI Account
                </span>
              </h1>

              <p
                className="
                  mt-6
                  max-w-xl
                  text-lg
                  leading-8
                  text-muted-foreground
                "
              >
                Start using advanced AI-powered visual speech
                recognition, transcript generation, and real-time
                lip reading technology.
              </p>

              <div className="mt-12 space-y-6">
                <div className="flex items-center gap-4">
                  <Brain className="h-6 w-6 text-primary" />

                  <div>
                    <p className="font-semibold">
                      AI-Powered Transcription
                    </p>

                    <p className="text-sm text-muted-foreground">
                      Generate transcripts from uploaded videos.
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  <Users className="h-6 w-6 text-primary" />

                  <div>
                    <p className="font-semibold">
                      Personal Dashboard
                    </p>

                    <p className="text-sm text-muted-foreground">
                      Access and manage all previous transcripts.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Mobile Header */}
            <div className="mx-auto w-full max-w-md">
              <div className="mb-8 text-center lg:hidden">
                <h1
                  className="
                    text-4xl
                    font-black
                    tracking-tight
                  "
                >
                  Create Account
                </h1>

                <p
                  className="
                    mt-4
                    text-muted-foreground
                  "
                >
                  Join LipSpeak AI and unlock powerful
                  speech recognition tools.
                </p>
              </div>

              <SignupForm />
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}