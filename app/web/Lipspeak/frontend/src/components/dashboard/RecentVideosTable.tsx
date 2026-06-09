/** RecentVideosTable.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Dashboard table displaying recent uploaded videos.
 * @date: 10 June 2026
 * @returns: Recent videos table component.
 *
 */


// Imports
import Link from "next/link";
import { ArrowUpRight } from "lucide-react";

import type {
  Video,
} from "@/types/video.types";


// Props Interface
interface RecentVideosTableProps {
  videos: Video[];
}


// Recent Videos Table Component
export function RecentVideosTable({
  videos,
}: Readonly<RecentVideosTableProps>) {
  // Empty State
  if (!videos.length) {
    return (
      <div
        className="
          glass-card
          ai-border
          rounded-3xl
          p-8
        "
      >
        <h2
          className="
            text-2xl
            font-bold
          "
        >
          Recent Videos
        </h2>

        <div
          className="
            mt-8
            rounded-2xl
            border
            border-dashed
            border-border
            p-10
            text-center
          "
        >
          <p className="text-muted-foreground">
            No videos uploaded yet.
          </p>

          <p
            className="
              mt-2
              text-sm
              text-muted-foreground
            "
          >
            Upload your first video to start
            generating transcripts.
          </p>
        </div>
      </div>
    );
  }

  // Render
  return (
    <div
      className="
        glass-card
        ai-border
        rounded-3xl
        p-6
      "
    >
      {/* Header */}
      <div
        className="
          flex
          items-center
          justify-between
        "
      >
        <div>
          <h2
            className="
              text-2xl
              font-bold
            "
          >
            Recent Videos
          </h2>

          <p
            className="
              mt-1
              text-sm
              text-muted-foreground
            "
          >
            Previously processed uploads.
          </p>
        </div>
      </div>

      {/* Desktop Table */}
      <div className="mt-6 hidden md:block">
        <div
          className="
            overflow-hidden
            rounded-2xl
            border
            border-border
          "
        >
          <table className="w-full">
            <thead>
              <tr
                className="
                  border-b
                  border-border
                "
              >
                <th className="px-6 py-4 text-left text-sm font-medium">
                  ID
                </th>

                <th className="px-6 py-4 text-left text-sm font-medium">
                  Transcript
                </th>

                <th className="px-6 py-4 text-left text-sm font-medium">
                  Date
                </th>

                <th className="px-6 py-4 text-right text-sm font-medium">
                  Action
                </th>
              </tr>
            </thead>

            <tbody>
              {videos.map((video) => (
                <tr
                  key={video.id}
                  className="
                    border-b
                    border-border
                    last:border-none
                  "
                >
                  <td className="px-6 py-4">
                    #{video.id}
                  </td>

                  <td className="px-6 py-4">
                    <p className="max-w-md truncate">
                      {video.transcript}
                    </p>
                  </td>

                  <td className="px-6 py-4 text-muted-foreground">
                    {new Date(
                      video.created_at,
                    ).toLocaleDateString()}
                  </td>

                  <td className="px-6 py-4 text-right">
                    <Link
                      href={`/dashboard/videos/${video.id}`}
                      className="
                        inline-flex
                        items-center
                        gap-2
                        text-primary
                        hover:underline
                      "
                    >
                      View

                      <ArrowUpRight
                        className="
                          h-4
                          w-4
                        "
                      />
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Mobile Cards */}
      <div className="mt-6 space-y-4 md:hidden">
        {videos.map((video) => (
          <div
            key={video.id}
            className="
              rounded-2xl
              border
              border-border
              p-4
            "
          >
            <div
              className="
                flex
                items-center
                justify-between
              "
            >
              <p className="font-semibold">
                #{video.id}
              </p>

              <p
                className="
                  text-xs
                  text-muted-foreground
                "
              >
                {new Date(
                  video.created_at,
                ).toLocaleDateString()}
              </p>
            </div>

            <p className="mt-3 text-sm">
              {video.transcript}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}