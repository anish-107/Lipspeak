/** AuthProvider.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Authentication provider responsible for restoring user sessions.
 * @date: 09 June 2026
 * @returns: Authentication provider component.
 *
 */


// Client Component
"use client";


// Imports
import { useEffect, useState } from "react";
import { authService } from "@/services/api/auth.service";
import { useAuthStore } from "@/store/auth.store";


// Types
interface AuthProviderProps {
  children: React.ReactNode;
}


// AuthProvider Component
export function AuthProvider({
  children,
}: AuthProviderProps) {
  // Store
  const setUser = useAuthStore(
    (state) => state.setUser,
  );

  const setToken = useAuthStore(
    (state) => state.setToken,
  );

  const logout = useAuthStore(
    (state) => state.logout,
  );

  // State
  const [isLoading, setIsLoading] =
    useState(true);

  // Effects
  useEffect(() => {
    // const initializeAuth = async () => {
    //   try {
    //     const token =
    //       localStorage.getItem(
    //         "access_token",
    //       );

    //     if (!token) {
    //       setIsLoading(false);
    //       return;
    //     }

    //     setToken(token);

    //     const user =
    //       await authService.getCurrentUser();

    //     // const user = {
    //     //   id: 1,
    //     //   username: "anish",
    //     //   name: "Anish Kumar",
    //     // };

    //     setUser(user);
        
    //   } catch (error) {
    //     console.error(
    //       "Failed to restore session:",
    //       error,
    //     );

    //     logout();
    //   } finally {
    //     setIsLoading(false);
    //   }
    // };

    const initializeAuth = async () => {
      const devMode =
        process.env.NEXT_PUBLIC_DEV_AUTH ===
        "true";
    
      if (devMode) {
        setToken("dev-token");
    
        setUser({
          id: 1,
          username: "anish",
          name: "Anish Kumar",
        });
    
        setIsLoading(false);
    
        return;
      }
    
      try {
        const token =
          localStorage.getItem(
            "access_token",
          );
    
        if (!token) {
          setIsLoading(false);
          return;
        }
        
        const user =
          await authService.getCurrentUser();
    
        setToken(token);
    
        setUser(user);
      } catch (error) {
        console.error(
          "Failed to restore session:",
          error,
        );
    
        localStorage.removeItem(
          "access_token",
        );
      } finally {
        setIsLoading(false);
      }
    };

    initializeAuth();
  }, [
    logout,
    setToken,
    setUser,
  ]);

  // Loading State
  if (isLoading) {
    return null;
  }

  // Render
  return children;
}