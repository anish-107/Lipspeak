/** DashboardOverview.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Main dashboard overview component with user and video statistics.
 * @date: 10 June 2026
 * @returns: Dashboard overview component.
 *
 */


// Client Component
"use client";


// Imports
import Link from "next/link";

import {
  BadgeCheck,
  Camera,
  FileVideo,
  Mic,
  Upload,
} from "lucide-react";

import { useAuthStore } from "@/store/auth.store";
import { useDashboard } from "@/hooks/useDashboard";

import { ProfileCard } from "@/components/dashboard/ProfileCard";
import { StatCard } from "@/components/dashboard/StatCard";
import { RecentVideosTable } from "@/components/dashboard/RecentVideosTable";


// Dashboard Overview Component
export function DashboardOverview() {
  /* ---------------------------------------------------------------------- */
  /*                                Store                                   */
  /* ---------------------------------------------------------------------- */

  const user = useAuthStore(
    (state) => state.user,
  );

  /* ---------------------------------------------------------------------- */
  /*                                Hook                                    */
  /* ---------------------------------------------------------------------- */

  const {
    videos,
    loading,
    error,
    totalVideos,
    latestTranscript,
  } = useDashboard();

  /* ---------------------------------------------------------------------- */
  /*                             Loading State                              */
  /* ---------------------------------------------------------------------- */

  if (loading) {
    return (
      <div
        className="
          flex
          min-h-[50vh]
          items-center
          justify-center
        "
      >
        <p className="text-muted-foreground">
          Loading dashboard...
        </p>
      </div>
    );
  }

  /* ---------------------------------------------------------------------- */
  /*                              Error State                               */
  /* ---------------------------------------------------------------------- */

  if (error) {
    return (
      <div
        className="
          rounded-3xl
          border
          border-red-500/20
          bg-red-500/10
          p-6
        "
      >
        <p className="text-red-500">
          {error}
        </p>
      </div>
    );
  }

  /* ---------------------------------------------------------------------- */
  /*                                Render                                  */
  /* ---------------------------------------------------------------------- */

  return (
    <section className="space-y-8">
      {/* Header */}
      <div>
        <h1
          className="
            text-4xl
            font-black
            tracking-tight
          "
        >
          Dashboard
        </h1>

        <p
          className="
            mt-3
            text-muted-foreground
          "
        >
          Manage transcripts, uploaded
          videos and real-time sessions.
        </p>
      </div>

      {/* Profile */}
      <ProfileCard user={user} />

      {/* Stats */}
      <div
        className="
          grid
          gap-6
          md:grid-cols-3
        "
      >
        <StatCard
          title="Total Videos"
          value={totalVideos}
          icon={FileVideo}
          description="Videos uploaded"
        />

        <StatCard
          title="Latest Transcript"
          value={latestTranscript}
          icon={Mic}
          description="Most recent result"
        />

        <StatCard
          title="Account Status"
          value="Active"
          icon={BadgeCheck}
          description="Authenticated session"
        />
      </div>

      {/* Quick Actions */}
      <div
        className="
          glass-card
          ai-border
          rounded-3xl
          p-6
        "
      >
        <h2
          className="
            text-2xl
            font-bold
          "
        >
          Quick Actions
        </h2>

        <p
          className="
            mt-2
            text-sm
            text-muted-foreground
          "
        >
          Start using LipSpeak AI.
        </p>

        <div
          className="
            mt-6
            grid
            gap-4
            md:grid-cols-2
          "
        >
          <Link
            href="/dashboard/pre-recorded"
            className="
              flex
              items-center
              gap-4
              rounded-2xl
              border
              border-border
              p-5
              transition-all
              hover:-translate-y-1
            "
          >
            <div
              className="
                flex
                h-12
                w-12
                items-center
                justify-center
                rounded-2xl
                bg-primary/10
              "
            >
              <Upload
                className="
                  h-5
                  w-5
                  text-primary
                "
              />
            </div>

            <div>
              <h3 className="font-semibold">
                Upload Video
              </h3>

              <p
                className="
                  text-sm
                  text-muted-foreground
                "
              >
                Analyse a pre-recorded video.
              </p>
            </div>
          </Link>

          <Link
            href="/dashboard/real-time"
            className="
              flex
              items-center
              gap-4
              rounded-2xl
              border
              border-border
              p-5
              transition-all
              hover:-translate-y-1
            "
          >
            <div
              className="
                flex
                h-12
                w-12
                items-center
                justify-center
                rounded-2xl
                bg-primary/10
              "
            >
              <Camera
                className="
                  h-5
                  w-5
                  text-primary
                "
              />
            </div>

            <div>
              <h3 className="font-semibold">
                Real-Time Session
              </h3>

              <p
                className="
                  text-sm
                  text-muted-foreground
                "
              >
                Start live lip-reading.
              </p>
            </div>
          </Link>
        </div>
      </div>

      {/* Videos */}
      <RecentVideosTable
        videos={videos}
      />
    </section>
  );
}