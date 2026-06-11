/** ProtectedRoute.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Route protection component for authenticated pages.
 * @date: 11 June 2026
 * @returns: Protected route wrapper.
 *
 */


// Client Component
"use client";


// Imports
import {
  useEffect,
} from "react";

import {
  useRouter,
} from "next/navigation";

import {
  useAuthStore,
} from "@/store/auth.store";


// Types
interface ProtectedRouteProps {
  children: React.ReactNode;
}


// Protected Route
export function ProtectedRoute({
  children,
}: ProtectedRouteProps) {
  /* ---------------------------------------------------------------------- */
  /*                                Hooks                                   */
  /* ---------------------------------------------------------------------- */

  const router =
    useRouter();

  const user =
    useAuthStore(
      (
        state,
      ) => state.user,
    );

  /* ---------------------------------------------------------------------- */
  /*                           Route Protection                             */
  /* ---------------------------------------------------------------------- */

  useEffect(() => {
    if (
      !user
    ) {
      router.replace(
        "/login",
      );
    }
  }, [
    user,
    router,
  ]);

  /* ---------------------------------------------------------------------- */
  /*                           Unauthenticated                              */
  /* ---------------------------------------------------------------------- */

  if (
    !user
  ) {
    return null;
  }

  /* ---------------------------------------------------------------------- */
  /*                                Return                                  */
  /* ---------------------------------------------------------------------- */

  return (
    <>
      {children}
    </>
  );
}