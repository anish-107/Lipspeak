/** video.types.ts
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Video related TypeScript types and interfaces.
 * @date: 09 June 2026
 * @returns: Video type definitions.
 *
 */


/* -------------------------------------------------------------------------- */
/*                                  Video                                     */
/* -------------------------------------------------------------------------- */

export interface Video {
  id: string;
  username: string;
  video_link: string;
  transcript: string;
  created_at: string;
}


/* -------------------------------------------------------------------------- */
/*                              Upload Request                                */
/* -------------------------------------------------------------------------- */

export interface UploadVideoRequest {
  file: File;
}


/* -------------------------------------------------------------------------- */
/*                              Upload Response                               */
/* -------------------------------------------------------------------------- */

export interface UploadVideoResponse {
  video_id: string;
  transcript: string;
}


/* -------------------------------------------------------------------------- */
/*                            Dashboard Response                              */
/* -------------------------------------------------------------------------- */

export interface VideosResponse {
  videos: Video[];
}