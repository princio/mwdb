import * as d3 from 'd3';

interface Datum001 {
  name: string;
  hidden: boolean;
  dga: number;
  type: 0|1;
  values: [number, number][];
}

interface SVG {
  svg: d3.Selection<SVGGElement, Datum001, HTMLElement, any>;
  x: d3.ScaleLinear<number, number, never>;
  y: d3.ScaleLinear<number, number, never>;
  data: Datum001[];
}


const plot_vline = (svg: SVG, th: number) => {
  
  svg.svg.selectAll<SVGTextElement, Datum001>('.vline_value')
    .transition()
    .duration(1000)
    .attr('x', svg.x(th))
    .attr('y', (d) => svg.y(d.values[th][1]))
    .text((d) => `${(d.values[th][1]).toFixed(2)}`);

  svg.svg.select('.vline')
    .transition()
    .duration(1000)
    .attr("x1", svg.x(th))
    .attr("y1", svg.y(0))
    .attr("x2", svg.x(th))
    .attr("y2", svg.y(1));

}


const plot_scatter =
    function(
        div_id: string, ths: number[],
        cms0: {n: number, p: number[], tn: number[], tp: [number[], number[], number[]]},
        cms1: {n: number, p: number[], tn: number[], tp: [number[], number[], number[]]}) : SVG {

  let data = [{}] as Datum001[];
  let data2 = [{}] as Datum001[];

  d3.select(`#${div_id}`).select('svg').remove();

  d3.selectAll(".checkbox_dga_class").on("change", (_event) => {
    const dga = +_event.target.value;
    
    area.selectAll(`.dga_selector_${dga}`)
      .transition()
      .duration(2000)
      .style("visibility", () => (_event.target.checked) ? "visible" : "hidden");
  });

  d3.selectAll(".checkbox_type_class").on("change", (_event) => {
    const type = +_event.target.value;
    
    area.selectAll(`.type_selector_${type}`)
      .transition()
      .duration(2000)
      .style("visibility", () => (_event.target.checked) ? "visible" : "hidden");
  });

  const margin = {top: 10, right: 100, bottom: 30, left: 30},
        width = 2 * (460 - margin.left - margin.right),
        height = 400 - margin.top - margin.bottom;

  const svg =
      d3.select<SVGGElement, Datum001>(`#${div_id}`)
          .append('svg')
          .attr('width', width + margin.left + margin.right)
          .attr('height', height + margin.top + margin.bottom)
          .append('g')
          .attr(
              'transform', 'translate(' + margin.left + ',' + margin.top + ')');

  svg.append('defs')
                 .append('svg:clipPath')
                 .attr('id', 'clip')
                 .append('svg:rect')
                 .attr('width', width)
                 .attr('height', height)
                 .attr('x', 0)
                 .attr('y', 0);

  var area = svg.append('g').attr('clip-path', 'url(#clip)')

  const dgas = ['dga_all', 'dga_0', 'dga_1', 'dga_2'];

  const myColor =
      d3.scaleOrdinal().domain(dgas).range(['blue', 'orange', 'red']);

  if (cms0 !== undefined) {
    data = dgas.map((dga, index_dga) => ({
                      name: dga,
                      dga: index_dga,
                      type: 0,
                      hidden: false,
                      values: ths.map((th, index) => ([
                                        index, cms0.tp[index_dga][index] / cms0.p[index_dga]
                                      ] as [number, number]))
                    }));
  }
  if (cms1 !== undefined) {
    data2 = dgas.map((dga, index_dga) => ({
                       name: dga,
                       dga: index_dga,
                       type: 1,
                       hidden: false,
                       values: ths.map((th, index) => ([
                                         index, cms1.tp[index_dga][index] / cms1.p[index_dga]
                                       ] as [number, number]))
                     }));
    data = [...data, ...data2];
  }

  const x = d3.scaleLinear().domain([0, 200]).range([0, width]);

  const xSVG = svg.append('g')
                   .attr('transform', 'translate(0,' + height + ')')
                   .call(d3.axisBottom(x));

  const y = d3.scaleLinear().domain([0, 1.02]).range([height, 0]);

  svg.append('g').call(d3.axisLeft(y));

  const areaGenerator = d3.area().x((d) => x(d[0])).y0(y(0)).y1((d) => y(d[1]));

  const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1]));

  area.selectAll('myLines')
      .data(data)
      .enter()
      .append('path')
      .attr('id', (d, dga) => `line_${d.dga}_${d.type}`)
      .attr('class', (d, dga) => `type_selector_${d.type} dga_selector_${d.dga} line_dga_type`)
      .attr('d', (d) => line(d.values))
      .attr('fill-opacity', .0)
      .attr('stroke', (d) => myColor(d.name) as string)
      .style('stroke-width', 1)
      .style('stroke-dasharray', (d, i) => [('0, 0'), ('3, 3')][d.type]);

  const x_pos = [
    (width) * 0.4,
    (width) * 0.6
  ];

  area.selectAll('vline_value')
      .data(data)
      .enter()
      .append('text')
      .attr('class', d => `vline_value type_selector_${d.type} dga_selector_${d.dga}`)
      .attr('x', x(50))
      .attr('y', (d) => y(d.values[50][1]))
      .style('fill', (d) => myColor(d.name) as string)
      .text((d) => `${(d.values[50][1]).toFixed(2)}`)
      .style('stroke-width', 0.5)
      .attr('text-anchor', 'center')
      .style('font-size', '10px');

  area.selectAll('myLegend')
      .data(data)
      .enter()
      .append('text')
      .attr('class', 'myLegend')
      .attr('class', (d) => `type_selector_${d.type} dga_selector_${d.dga}`)
      .attr('x', d => x_pos[d.type])
      .attr('y', (d) => y(d.values[Math.floor(x.invert(x_pos[d.type]))][1]))
      .text((d) => `${d.name}/${d.type}`)
      .style('fill', (d) => myColor(d.name) as string)
      .attr('text-anchor', 'center')
      .style('font-size', '10px');
    
  area
    .append("line")
    .attr('class', 'vline')
    .attr("x1", x(50))
    .attr("y1", y(0))
    .attr("x2", x(50))
    .attr("y2", y(1))
    .style("stroke-width", 1)
    .style("stroke", "grey")
    .style("fill", "none");

  // BRUSH ////////////////////////

  var brush = d3.brushX<Datum001>()
                  .extent([[0, 0], [width, height]])
                  .on('end', updateChart);

  area.append('g').attr('class', 'brush').call(brush);


  var idleTimeout: number|null;
  function idled() {
    idleTimeout = null;
  }

  function updateChart(_event: any) {
    const extent = _event.selection

    if (!extent) {
      if (!idleTimeout) return idleTimeout = setTimeout(idled, 350);
      x.domain([4, 8])
    }
    else {
      x.domain([x.invert(extent[0]), x.invert(extent[1])]);
      svg.select<SVGGElement>('.brush').call(brush.move, null);
    }

    xSVG.transition().duration(1000).call(d3.axisBottom(x))

    area.selectAll<SVGPathElement, Datum001>('.line_dga_type')
        .transition()
        .duration(1000)
        .attr('d', (d) => line(d.values));

    if (extent) {
      
      const x_pos = [
        x.range()[0] + (x.range()[1] - x.range()[0]) * 0.4,
        x.range()[0] + (x.range()[1] - x.range()[0]) * 0.6
      ];

      area.selectAll<SVGTextElement, Datum001>('.myLegend')
          .transition()
          .duration(1000)
          .attr('x', d => x_pos[d.type])
          .attr('y', d => y(d.values[Math.floor(x.invert(x_pos[d.type]))][1]));
    }
  }

  svg.on('dblclick', () => {
    x.domain([0, 200]);

    xSVG.transition().call(d3.axisBottom(x));

    area.selectAll<SVGPathElement, Datum001>('.line_dga_type')
        .transition()
        .duration(200)
        .attr('d', (d) => line(d.values));
      
    const x_pos = [
      x.range()[0] + (x.range()[1] - x.range()[0]) * 0.4,
      x.range()[0] + (x.range()[1] - x.range()[0]) * 0.6
    ]

    area.selectAll<SVGTextElement, Datum001>('.myLegend')
        .transition()
        .duration(200)
        .attr('x', d => x_pos[d.type])
        .attr('y', (d) => y(d.values[Math.floor(x.invert(x_pos[d.type]))][1]));
  });

  // BRUSH ////////////////////////

  return {
    svg,
    x,
    y,
    data
  };
}

export {
  plot_vline,
  plot_scatter
};