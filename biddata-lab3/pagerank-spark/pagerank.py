#!/usr/bin/env python3
"""PageRank on Spark (pseudo-distributed HDFS input, 10 iterations)."""
import sys

from pyspark import SparkConf, SparkContext

DAMPING = 0.85
ITERATIONS = 10


def parse(line):
    parts = line.strip().split("\t")
    page = parts[0]
    links = parts[1].split(",") if len(parts) > 1 and parts[1] else []
    return page, links


def main():
    if len(sys.argv) != 3:
        print("Usage: pagerank.py <input> <output>", file=sys.stderr)
        sys.exit(2)
    input_path, output_path = sys.argv[1], sys.argv[2]

    sc = SparkContext(conf=SparkConf().setAppName("PageRank-Spark"))
    sc.setLogLevel("WARN")

    graph = sc.textFile(input_path).map(parse).cache()
    n = graph.count()
    print("Total pages: %d" % n, flush=True)

    ranks = graph.mapValues(lambda links: 1.0).cache()
    for i in range(1, ITERATIONS + 1):
        joined = graph.join(ranks)
        contributions = joined.flatMap(
            lambda kv: [(t, kv[1][1] / len(kv[1][0])) for t in kv[1][0]]
        )
        dangling = joined.filter(lambda kv: len(kv[1][0]) == 0) \
                         .map(lambda kv: kv[1][1]).sum()
        base = (1.0 - DAMPING) + DAMPING * dangling / n

        # union with all page keys so that pages without in-links are kept
        new_ranks = graph.keys().map(lambda p: (p, 0.0)).union(contributions)
        ranks = new_ranks.reduceByKey(lambda a, b: a + b) \
                         .mapValues(lambda s: base + DAMPING * s).cache()
        print("Iteration %d/%d finished" % (i, ITERATIONS), flush=True)

    result = ranks.sortBy(lambda kv: -kv[1]) \
                  .map(lambda kv: "(%s,%.10f)" % (kv[0], kv[1]))

    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    out = sc._jvm.org.apache.hadoop.fs.Path(output_path)
    if fs.exists(out):
        fs.delete(out, True)
    result.coalesce(1).saveAsTextFile(output_path)

    print("\n===== PageRank result after %d iterations =====" % ITERATIONS, flush=True)
    for line in result.collect():
        print(line, flush=True)

    sc.stop()


if __name__ == "__main__":
    main()
