/** ProfileCard.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Dashboard profile information card.
 * @date: 10 June 2026
 * @returns: Profile card component.
 *
 */


// Imports
import { CalendarDays, User } from "lucide-react";

import type {
  User as UserType,
} from "@/types/auth.types";


// Props Interface
interface ProfileCardProps {
  user: UserType | null;
}


// Profile Card Component
export function ProfileCard({
  user,
}: Readonly<ProfileCardProps>) {
  const avatar =
    user?.name?.charAt(0).toUpperCase() ??
    "U";

  return (
    <div
      className="
        glass-card
        ai-border
        rounded-3xl
        p-6
      "
    >
      <div
        className="
          flex
          flex-col
          gap-6
          sm:flex-row
          sm:items-center
        "
      >
        {/* Avatar */}
        <div
          className="
            flex
            h-20
            w-20
            items-center
            justify-center
            rounded-full
            bg-linear-to-r
            from-indigo-500
            via-cyan-500
            to-purple-500
            text-3xl
            font-black
            text-white
          "
        >
          {avatar}
        </div>

        {/* User Info */}
        <div className="flex-1">
          <h2
            className="
              text-2xl
              font-black
            "
          >
            {user?.name ?? "User"}
          </h2>

          <p
            className="
              mt-1
              text-muted-foreground
            "
          >
            @{user?.username ?? "guest"}
          </p>

          <div
            className="
              mt-4
              flex
              flex-wrap
              gap-4
            "
          >
            <div
              className="
                flex
                items-center
                gap-2
                text-sm
                text-muted-foreground
              "
            >
              <User className="h-4 w-4" />

              Registered User
            </div>

            <div
              className="
                flex
                items-center
                gap-2
                text-sm
                text-muted-foreground
              "
            >
              <CalendarDays className="h-4 w-4" />

              LipSpeak AI Member
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}