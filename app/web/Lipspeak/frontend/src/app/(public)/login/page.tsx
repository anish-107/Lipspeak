/** page.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Public login page for user authentication.
 * @date: 10 May 2026
 * @returns: Login page component.
 *
 */


// Imports
import { Navbar } from "@/components/common/Navbar";
import { Footer } from "@/components/common/Footer";
import { LoginForm } from "@/components/auth/LoginForm";
import { BackgroundBeams } from "@/components/ui/aceternity/background-beams";
import { Brain, ShieldCheck, Sparkles } from "lucide-react";


// Login Page Component
export default function LoginPage() {
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
                AI-Powered Authentication
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
                Welcome Back To
                <span
                  className="
                    block
                    gradient-text
                  "
                >
                  LipSpeak AI
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
                Access real-time lip reading, video transcription,
                transcript history, and advanced AI-powered
                speech recognition tools.
              </p>

              <div className="mt-12 space-y-6">
                <div className="flex items-center gap-4">
                  <Brain className="h-6 w-6 text-primary" />

                  <div>
                    <p className="font-semibold">
                      Real-Time Recognition
                    </p>

                    <p className="text-sm text-muted-foreground">
                      Live AI transcription from webcam streams.
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  <ShieldCheck className="h-6 w-6 text-primary" />

                  <div>
                    <p className="font-semibold">
                      Secure Dashboard
                    </p>

                    <p className="text-sm text-muted-foreground">
                      Access your transcript history anytime.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Form */}
            <div className="mx-auto w-full max-w-md">
              <div className="mb-8 text-center lg:hidden">
                <h1
                  className="
                    text-4xl
                    font-black
                    tracking-tight
                  "
                >
                  Welcome Back
                </h1>

                <p
                  className="
                    mt-4
                    text-muted-foreground
                  "
                >
                  Sign in to continue using LipSpeak AI.
                </p>
              </div>

              <LoginForm />
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}