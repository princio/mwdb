<script lang="ts">
import axios from "axios";
import { defineComponent } from "vue";
import { plot_scatter, plot_vline } from "../assets/plotter";
import * as d3 from "d3";

type ReplaceValue = ["llr" | "nx", [number, number], [string, string]];

interface Configs {
  models: string[];
  wsize: number[];
  windowing: string[];
  top10m: [number, number][];
  replace_values: ReplaceValue[];
}

interface Config {
  model: "none" | "icann" | "tld" | "private" | "nx";
  wsize: 100 | 500 | 2500;
  windowing: "req" | "res" | "both";
  top10m: [number, number];
  replace_values: ReplaceValue;
}

interface CMS {
  ths: number[];
  cms: {
    n: number;
    p: number[];
    tn: number[];
    fp: number[];
    fn: [number[], number[], number[]];
    tp: [number[], number[], number[]];
  };
}

const rv_type = (config: Config) => (config.model === "nx" ? "nx" : "llr");

export default {
  name: "ConfigBar",
  emits: {},
  props: {
    cms_0?: CMS,
    cms_1: CMS
  },
  data: function () {
    return {};
  },
};
</script>

<template>
  <div class="configs">
    <div v-if="cms_0.ths">
      <table>
        <thead>
          <tr>
            <th></th>
            <th>N</th>
            <th>P</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>N</td>
            <td>{{ cms_0.cms.tn[th_chosen] }}</td>
            <td>{{ cms_0.cms.fp[th_chosen] }}</td>
          </tr>
          <tr>
            <td>P</td>
            <td>
              {{
                cms_0.cms.fn[0][th_chosen] +
                cms_0.cms.fn[1][th_chosen] +
                cms_0.cms.fn[2][th_chosen]
              }}
            </td>
            <td>
              {{
                cms_0.cms.tp[0][th_chosen] +
                cms_0.cms.tp[1][th_chosen] +
                cms_0.cms.tp[2][th_chosen]
              }}
            </td>
          </tr>
        </tbody>
      </table>
      <table v-for="dga of [0, 1, 2]" :key="`cm_0_${dga}`">
        <tbody>
          <tr>
            <td>{{ cms_0.cms.tn[th_chosen] }}</td>
            <td>{{ cms_0.cms.fp[th_chosen] }}</td>
          </tr>
          <tr>
            <td>{{ cms_0.cms.fn[dga][th_chosen] }}</td>
            <td>{{ cms_0.cms.tp[dga][th_chosen] }}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <br />
    {{ integrals1 }}
  </div>
</template>

<style scoped>
.plots {
  display: flex;
  flex-direction: row;
  justify-content: center;
  width: 100%;
}
.row {
  display: flex;
  flex-direction: row;
  justify-content: center;
}
.plots > div {
  width: 100%;
}
.active {
  background-color: lightskyblue;
}
.boldd {
  font-weight: bold;
  font-size: 16px;
}
.center {
  text-align: center;
  margin: auto;
}
#app > div.configs > table > tbody {
  text-align: left;
}
table > tbody > tr > td,
th {
  text-align: left;
}
</style>
