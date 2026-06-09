/** StatCard.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Reusable dashboard statistic card component.
 * @date: 10 June 2026
 * @returns: Statistic card component.
 *
 */


// Imports
import type { LucideIcon } from "lucide-react";


// Props Interface
interface StatCardProps {
  title: string;

  value: string | number;

  icon: LucideIcon;

  description?: string;
}


// Stat Card Component
export function StatCard({
  title,
  value,
  icon: Icon,
  description,
}: Readonly<StatCardProps>) {
  // Render
  return (
    <div
      className="
        glass-card
        ai-border
        rounded-3xl
        p-6
        transition-all
        duration-300
        hover:-translate-y-1
      "
    >
      <div
        className="
          flex
          items-start
          justify-between
        "
      >
        <div>
          <p
            className="
              text-sm
              text-muted-foreground
            "
          >
            {title}
          </p>

          <h3
            className="
              mt-3
              text-3xl
              font-black
              tracking-tight
            "
          >
            {value}
          </h3>

          {description && (
            <p
              className="
                mt-2
                text-sm
                text-muted-foreground
              "
            >
              {description}
            </p>
          )}
        </div>

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
          <Icon
            className="
              h-6
              w-6
              text-primary
            "
          />
        </div>
      </div>
    </div>
  );
}