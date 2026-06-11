/** AuthProvider.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Authentication provider responsible for restoring user sessions.
 * @date: 11 June 2026
 * @returns: Authentication provider component.
 *
 */


// Client Component
"use client";


// Imports
import {
  useEffect,
  useState,
} from "react";

import {
  authService,
} from "@/services/api/auth.service";

import {
  useAuthStore,
} from "@/store/auth.store";


// Types
interface AuthProviderProps {
  children: React.ReactNode;
}


// AuthProvider Component
export function AuthProvider({
  children,
}: AuthProviderProps) {
  /* ---------------------------------------------------------------------- */
  /*                                Store                                   */
  /* ---------------------------------------------------------------------- */

  const setUser =
    useAuthStore(
      (
        state,
      ) => state.setUser,
    );

  const setToken =
    useAuthStore(
      (
        state,
      ) => state.setToken,
    );

  const logout =
    useAuthStore(
      (
        state,
      ) => state.logout,
    );

  /* ---------------------------------------------------------------------- */
  /*                                State                                   */
  /* ---------------------------------------------------------------------- */

  const [
    isLoading,
    setIsLoading,
  ] = useState(
    true,
  );

  /* ---------------------------------------------------------------------- */
  /*                           Session Restore                              */
  /* ---------------------------------------------------------------------- */

  useEffect(() => {
    const initializeAuth =
      async () => {
        try {
          const token =
            localStorage.getItem(
              "access_token",
            );

          if (
            !token
          ) {
            return;
          }

          setToken(
            token,
          );

          const user =
            await authService.getCurrentUser();

          setUser(
            user,
          );

        } catch (error) {
          console.error(
            "Failed to restore session:",
            error,
          );

          logout();

        } finally {
          setIsLoading(
            false,
          );
        }
      };

    void initializeAuth();
  }, [
    logout,
    setToken,
    setUser,
  ]);

  /* ---------------------------------------------------------------------- */
  /*                               Loading                                  */
  /* ---------------------------------------------------------------------- */

  if (
    isLoading
  ) {
    return null;
  }

  /* ---------------------------------------------------------------------- */
  /*                                Render                                  */
  /* ---------------------------------------------------------------------- */

  return children;
}