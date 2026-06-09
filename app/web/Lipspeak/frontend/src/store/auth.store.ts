/** auth.store.ts
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Global authentication state management using Zustand.
 * @date: 09 June 2026
 * @returns: Authentication store.
 *
 */


// Imports
import { create } from "zustand";
import { AuthState, User } from "@/types/auth.types";


// Store Types
interface AuthStore extends AuthState {
  setToken: (token: string | null) => void;

  setUser: (user: User | null) => void;

  login: (token: string, user: User) => void;

  logout: () => void;
}


// Auth Store
export const useAuthStore = create<AuthStore>((set) => ({
  user: null,

  token: null,

  isAuthenticated: false,


  /* ------------------------------------------------------------------------ */
  /*                                Set Token                                 */
  /* ------------------------------------------------------------------------ */

  setToken: (token) =>
    set({
      token,
      isAuthenticated: !!token,
    }),


  /* ------------------------------------------------------------------------ */
  /*                                 Set User                                 */
  /* ------------------------------------------------------------------------ */

  setUser: (user) =>
    set({
      user,
    }),


  /* ------------------------------------------------------------------------ */
  /*                                   Login                                  */
  /* ------------------------------------------------------------------------ */

  login: (token, user) => {
    localStorage.setItem("access_token", token);

    set({
      token,
      user,
      isAuthenticated: true,
    });
  },


  /* ------------------------------------------------------------------------ */
  /*                                  Logout                                  */
  /* ------------------------------------------------------------------------ */

  logout: () => {
    localStorage.removeItem("access_token");

    set({
      token: null,
      user: null,
      isAuthenticated: false,
    });
  },
}));