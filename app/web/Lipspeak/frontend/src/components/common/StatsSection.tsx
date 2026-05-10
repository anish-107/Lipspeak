// src/components/common/StatsSection.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Statistics section for landing page metrics.
 * @date: 10 May 2026
 * @returns: Stats section component.
 */

const stats = [
  {
    label: "Inference Accuracy",
    value: "97%",
  },
  {
    label: "Prediction Speed",
    value: "40ms",
  },
  {
    label: "AI Models",
    value: "12+",
  },
  {
    label: "Video Support",
    value: "4K",
  },
];

export function StatsSection() {
  return (
    <section className="px-6 py-24">
      <div className="mx-auto max-w-7xl rounded-[40px] border border-white/10 bg-gradient-to-br from-zinc-900 to-black p-12">
        <div className="grid md:grid-cols-2 xl:grid-cols-4 gap-10">
          {stats.map((stat) => (
            <div key={stat.label}>
              <h3 className="text-5xl font-black bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
                {stat.value}
              </h3>

              <p className="mt-4 text-zinc-400">
                {stat.label}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}