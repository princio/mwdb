import * as d3 from "d3";

export type TPTN = {
  p: boolean;
  n: boolean;
};
export type Metric = {
  ac: boolean;
  pr: boolean;
  re: boolean;
  f1: boolean;
  roc: boolean;
};

interface Datum001 {
  name: string;
  hidden: boolean;
  dga: number;
  type: 0 | 1;
  metric: keyof Metric;
  tp: keyof TPTN;
  values: [number, number][];
}

interface PlotConfig {
  tp: TPTN;
  metric: Metric;
  dga: [boolean, boolean, boolean];
  plots: [boolean, boolean];
}

interface SVG {
  svg: d3.Selection<SVGGElement, Datum001, HTMLElement, any>;
  x: d3.ScaleLinear<number, number, never>;
  y: d3.ScaleLinear<number, number, never>;
  data: Datum001[];
}

const plot_vline = (svg: SVG, th: number) => {
  svg.svg
    .selectAll<SVGTextElement, Datum001>(".vline_value")
    .transition()
    .duration(1000)
    .attr("x", svg.x(th))
    .attr("y", (d) => svg.y(d.values[th][1]))
    .text((d) => `${d.values[th][1].toFixed(2)}`);

  svg.svg
    .select(".vline")
    .transition()
    .duration(1000)
    .attr("x1", svg.x(th))
    .attr("y1", svg.y(0))
    .attr("x2", svg.x(th))
    .attr("y2", svg.y(1));
};

const plot_config: PlotConfig = {
  tp: {
    p: true,
    n: false,
  },
  metric: {
    ac: true,
    pr: false,
    re: false,
    f1: false,
    roc: false,
  },
  dga: [true, false, false],
  plots: [true, true],
};

interface CM {
  n: number;
  p: number[];
  tn: number[];
  fp: number[];
  fn: [number[], number[], number[]];
  tp: [number[], number[], number[]];
}

const plot_scatter = function (
  div_id: string,
  ths: number[],
  cms0: CM,
  cms1: CM
): SVG {
  let data = [{}] as Datum001[];
  let data2 = [{}] as Datum001[];
  const checkbox_invert = false;

  d3.select(`#${div_id}`).select("svg").remove();

  // d3.selectAll(".checkbox_dga_class").on("click", (_event: PointerEvent) => {
  //   checkbox_invert = _event.shiftKey;

  //   if (checkbox_invert)
  //     d3.selectAll(".checkbox_dga_class")
  //       .property('checked', false);

  // });

  const update_data = () => {
    data.forEach((d) => {
      d.hidden = !(
        plot_config.dga[d.dga] &&
        (plot_config.tp[d.tp] as boolean) &&
        plot_config.metric[d.metric] &&
        plot_config.plots[d.type]
      );
    });

    console.log(plot_config);

    area
      .selectAll<any, Datum001>(`.plots_selector`)
      .transition()
      .duration(2000)
      .style("visibility", (d) => (!d.hidden ? "visible" : "hidden"));
  };

  d3.selectAll(".checkbox_tp_n").on("change", (_event) => {
    const value = _event.target.checked;

    plot_config.tp.n = value;

    update_data();
  });

  d3.selectAll(".checkbox_tp_p").on("change", (_event) => {
    const value = _event.target.checked;

    plot_config.tp.p = value;

    update_data();
  });

  d3.selectAll(".select_metric").on("change", (_event) => {
    const metric_value = _event.target.value as keyof Metric;

    for (const metric in plot_config.metric) {
      plot_config.metric[metric as keyof Metric] = metric_value === metric;
    }

    update_data();
  });

  d3.selectAll(".checkbox_dga_class").on("change", (_event) => {
    const dga = _event.target.value;
    const checked = _event.target.checked;

    plot_config.dga[dga] = checked;

    update_data();
  });

  d3.selectAll(".checkbox_type_class").on("change", (_event) => {
    const plot = _event.target.value;
    const checked = _event.target.checked;

    plot_config.plots[plot] = checked;

    update_data();
  });

  const margin = { top: 10, right: 100, bottom: 30, left: 30 },
    width = 2 * (600 - margin.left - margin.right),
    height = 400 - margin.top - margin.bottom;

  const svg = d3
    .select<SVGGElement, Datum001>(`#${div_id}`)
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  svg
    .append("defs")
    .append("svg:clipPath")
    .attr("id", "clip")
    .append("svg:rect")
    .attr("width", width)
    .attr("height", height)
    .attr("x", 0)
    .attr("y", 0);

  const area = svg.append("g").attr("clip-path", "url(#clip)");

  const dgas = ["all", "dga_1", "dga_2"];

  const myColor = d3
    .scaleOrdinal()
    .domain(dgas)
    .range(["gray", "blue", "orange", "red"]);

  const get_data = (
    cms: CM,
    metric: string,
    tp: string,
    index: number,
    dga: number
  ) => {
    let value;

    if (metric === "pr") {
      // Precision = TP/TP+FP
      if (tp === "n") {
        value = cms.tn[index] / (cms.tn[index] + cms.fn[dga][index]);
      } else {
        value = cms.tp[dga][index] / (cms.tp[dga][index] + cms.fp[index]);
      }
    } else if (metric === "ac") {
      // Accuracy = TP+TN/TP+FP+FN+TN
      value = (cms.tn[index] + cms.tp[dga][index]) / (cms.n + cms.p[dga]);
    } else if (metric === "roc") {
      // TPR (True Positive Rate) = Recall_P
      // FPR (False Positive Rate) = FP / N
      const ac = (cms.tn[index] + cms.tp[dga][index]) / (cms.n + cms.p[dga]);
      const fpr = cms.fp[index] / cms.n;
      value = ac + fpr;
    } else if (metric === "re") {
      // Recall = TP/P
      if (tp === "n") {
        value = cms.tn[index] / cms.n;
      } else {
        value = cms.tp[dga][index] / cms.p[dga];
      }
    } else if (metric === "f1") {
      if (tp === "n") {
        const pr = cms.tn[index] / (cms.tn[index] + cms.fn[dga][index]);
        const re = cms.tn[index] / (cms.tn[index] + cms.fp[index]);
        value = (2 * pr * re) / (pr + re);
      } else {
        const pr = cms.tp[dga][index] / (cms.tp[dga][index] + cms.fp[index]);
        const re =
          cms.tp[dga][index] / (cms.tp[dga][index] + cms.fn[dga][index]);
        value = (2 * pr * re) / (pr + re);
      }
    } else {
      if (tp === "n") {
        value = cms.tn[index] / cms.n;
      } else {
        value = cms.tp[dga][index] / cms.p[dga];
      }
    }

    return [index, isNaN(value) ? 0 : value] as [number, number];
  };

  if (cms0 !== undefined) {
    data = ["p", "n"].flatMap((tp: string) =>
      ["pr", "re", "ac", "f1", "roc"].flatMap((metric: string) =>
        dgas.map((dga, index_dga) => ({
          name: dga,
          dga: index_dga,
          type: 0,
          hidden: true,
          metric: metric as keyof Metric,
          tp: tp as keyof TPTN,
          values: ths.map((_th, index) =>
            get_data(cms0, metric, tp, index, index_dga)
          ),
        }))
      )
    );
  }
  if (cms1 !== undefined) {
    data2 = ["p", "n"].flatMap((tp: string) =>
      ["pr", "re", "ac", "f1"].flatMap((metric: string) =>
        dgas.map((dga, index_dga) => ({
          name: dga,
          dga: index_dga,
          type: 1,
          hidden: false,
          metric: metric as keyof Metric,
          tp: tp as keyof TPTN,
          values: ths.map((_th, index) =>
            get_data(cms1, metric, tp, index, index_dga)
          ),
        }))
      )
    );
    data = [...data, ...data2];
  }

  const x = d3.scaleLinear().domain([0, 200]).range([0, width]);

  const xSVG = svg
    .append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x));

  const y = d3.scaleLinear().domain([0, 1.02]).range([height, 0]);

  svg.append("g").call(d3.axisLeft(y));

  const areaGenerator = d3
    .area()
    .x((d) => x(d[0]))
    .y0(y(0))
    .y1((d) => y(d[1]));

  const line = d3
    .line()
    .x((d) => x(d[0]))
    .y((d) => y(d[1]));

  area
    .selectAll("myLines")
    .data(data)
    .enter()
    .append("path")
    .attr("id", (d) => `line_${d.dga}_${d.type}_${d.metric}_${d.tp}`)
    .attr("class", (d) => {
      return `plots_selector type_selector_${d.type} dga_selector_${d.dga} line_dga_type metric_selector_${d.metric} tp_selector_${d.tp}`;
    })
    .attr("d", (d) => {
      return line(d.values);
    })
    .attr("fill-opacity", 0.0)
    .attr("stroke", (d) => {
      return { n: "blue", p: "red" }[d.tp];
    })
    .style("stroke-width", (d) => myColor(d.name) as string)
    .style("stroke-dasharray", (d) => ["0, 0", "3, 3"][d.type])
    .style("visibility", (d) => (!d.hidden ? "visible" : "hidden"));

  const x_pos = [width * 0.4, width * 0.6];

  area
    .selectAll("vline_value")
    .data(data)
    .enter()
    .append("text")
    .attr(
      "class",
      (d) => `vline_value type_selector_${d.type} dga_selector_${d.dga}`
    )
    .attr("x", x(50))
    .attr("y", (d) => y(d.values[50][1]))
    .style("fill", (d) => myColor(d.name) as string)
    .text((d) => `${d.values[50][1].toFixed(2)}`)
    .style("stroke-width", 0.5)
    .attr("text-anchor", "center")
    .style("font-size", "10px")
    .style("visibility", (d) => (!d.hidden ? "visible" : "hidden"));

  area
    .selectAll("myLegend")
    .data(data)
    .enter()
    .append("text")
    .attr("class", "myLegend")
    .attr("class", (d) => `type_selector_${d.type} dga_selector_${d.dga}`)
    .attr("x", (d) => x_pos[d.type])
    .attr("y", (d) => y(d.values[Math.floor(x.invert(x_pos[d.type]))][1]))
    .text((d) => `${d.name}/${d.type}/${d.dga}/${d.tp}`)
    .style("fill", (d) => myColor(d.name) as string)
    .attr("text-anchor", "center")
    .style("font-size", "10px")
    .style("visibility", (d) => (!d.hidden ? "visible" : "hidden"));

  area
    .append("line")
    .attr("class", "vline")
    .attr("x1", x(50))
    .attr("y1", y(0))
    .attr("x2", x(50))
    .attr("y2", y(1))
    .style("stroke-width", 1)
    .style("stroke", "grey")
    .style("fill", "none");

  // BRUSH ////////////////////////

  const brush = d3
    .brushX<Datum001>()
    .extent([
      [0, 0],
      [width, height],
    ])
    .on("end", updateChart);

  area.append("g").attr("class", "brush").call(brush);

  let idleTimeout: number | null;
  function idled() {
    idleTimeout = null;
  }

  function updateChart(_event: any) {
    const extent = _event.selection;

    if (!extent) {
      if (!idleTimeout) return (idleTimeout = setTimeout(idled, 350));
      x.domain([4, 8]);
    } else {
      x.domain([x.invert(extent[0]), x.invert(extent[1])]);
      svg.select<SVGGElement>(".brush").call(brush.move, null);
    }

    xSVG.transition().duration(1000).call(d3.axisBottom(x));

    area
      .selectAll<SVGPathElement, Datum001>(".line_dga_type")
      .transition()
      .duration(1000)
      .attr("d", (d) => line(d.values));

    if (extent) {
      const x_pos = [
        x.range()[0] + (x.range()[1] - x.range()[0]) * 0.4,
        x.range()[0] + (x.range()[1] - x.range()[0]) * 0.6,
      ];

      area
        .selectAll<SVGTextElement, Datum001>(".myLegend")
        .transition()
        .duration(1000)
        .attr("x", (d) => x_pos[d.type])
        .attr("y", (d) => y(d.values[Math.floor(x.invert(x_pos[d.type]))][1]));
    }
  }

  svg.on("dblclick", () => {
    x.domain([0, 200]);

    xSVG.transition().call(d3.axisBottom(x));

    area
      .selectAll<SVGPathElement, Datum001>(".line_dga_type")
      .transition()
      .duration(200)
      .attr("d", (d) => line(d.values));

    const x_pos = [
      x.range()[0] + (x.range()[1] - x.range()[0]) * 0.4,
      x.range()[0] + (x.range()[1] - x.range()[0]) * 0.6,
    ];

    area
      .selectAll<SVGTextElement, Datum001>(".myLegend")
      .transition()
      .duration(200)
      .attr("x", (d) => x_pos[d.type])
      .attr("y", (d) => y(d.values[Math.floor(x.invert(x_pos[d.type]))][1]));
  });

  // BRUSH ////////////////////////

  return {
    svg,
    x,
    y,
    data,
  };
};

export { plot_vline, plot_scatter };
