import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

/**
 * PageRank on Hadoop MapReduce (pseudo-distributed mode).
 * Input : "page \t target1,target2,..."
 * Output: "(page,0.xxxxxxxxxx)" with 10 decimal places, sorted by PR descending.
 */
public class PageRank {

    private static final double DAMPING = 0.85;
    private static final int ITERATIONS = 10;

    /** Initialization: page -> 1.0, keep the adjacency list. */
    public static class InitMapper extends Mapper<LongWritable, Text, Text, Text> {
        @Override
        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty()) {
                return;
            }
            String[] parts = line.split("\t", -1);
            String page = parts[0];
            String links = parts.length > 1 ? parts[1] : "";
            context.write(new Text(page), new Text("1.0\t" + links));
        }
    }

    /** Distribute the current PR along outgoing links, and carry the graph structure. */
    public static class IterMapper extends Mapper<LongWritable, Text, Text, Text> {
        private String[] allPages;

        @Override
        protected void setup(Context context) {
            allPages = context.getConfiguration().get("pagerank.pages").split(",");
        }

        @Override
        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String[] f = value.toString().split("\t", -1);
            if (f.length < 2) {
                return;
            }
            String page = f[0];
            double pr = Double.parseDouble(f[1]);
            String links = f.length > 2 ? f[2] : "";

            if (links.isEmpty()) {
                // Dangling page: spread its PR evenly over all pages.
                double share = pr / allPages.length;
                for (String p : allPages) {
                    context.write(new Text(p), new Text("#" + share));
                }
                context.write(new Text(page), new Text("N\t"));
            } else {
                String[] targets = links.split(",");
                double share = pr / targets.length;
                for (String t : targets) {
                    context.write(new Text(t), new Text("#" + share));
                }
                context.write(new Text(page), new Text("N\t" + links));
            }
        }
    }

    /** newPR = (1 - d) + d * sum(contributions). */
    public static class IterReducer extends Reducer<Text, Text, Text, Text> {
        @Override
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            double sum = 0.0;
            String links = "";
            for (Text v : values) {
                String s = v.toString();
                if (s.startsWith("#")) {
                    sum += Double.parseDouble(s.substring(1));
                } else if (s.startsWith("N\t")) {
                    links = s.substring(2);
                }
            }
            double pr = (1.0 - DAMPING) + DAMPING * sum;
            context.write(key, new Text(String.format(Locale.US, "%.12f", pr) + "\t" + links));
        }
    }

    /** Final job: sort by PR descending and format the output. */
    public static class SortMapper extends Mapper<LongWritable, Text, DoubleWritable, Text> {
        @Override
        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String[] f = value.toString().split("\t");
            if (f.length >= 2) {
                context.write(new DoubleWritable(Double.parseDouble(f[1])), new Text(f[0]));
            }
        }
    }

    public static class DescDoubleComparator extends WritableComparator {
        protected DescDoubleComparator() {
            super(DoubleWritable.class, true);
        }

        @Override
        public int compare(WritableComparable a, WritableComparable b) {
            DoubleWritable x = (DoubleWritable) a;
            DoubleWritable y = (DoubleWritable) b;
            return -Double.compare(x.get(), y.get());
        }
    }

    public static class SortReducer extends Reducer<DoubleWritable, Text, Text, NullWritable> {
        @Override
        public void reduce(DoubleWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            for (Text page : values) {
                String line = "(" + page.toString() + "," +
                        String.format(Locale.US, "%.10f", key.get()) + ")";
                context.write(new Text(line), NullWritable.get());
            }
        }
    }

    private static List<String> readLines(FileSystem fs, Path dir) throws IOException {
        List<String> lines = new ArrayList<>();
        FileStatus[] files = fs.globStatus(new Path(dir, "part-*"));
        if (files != null) {
            Arrays.sort(files, Comparator.comparing(f -> f.getPath().getName()));
            for (FileStatus st : files) {
                try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(st.getPath())))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        if (!line.trim().isEmpty()) {
                            lines.add(line);
                        }
                    }
                }
            }
        }
        return lines;
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: PageRank <input> <output>");
            System.exit(2);
        }
        Configuration conf = new Configuration();
        Path input = new Path(args[0]);
        Path output = new Path(args[1]);
        FileSystem fs = output.getFileSystem(conf);
        if (fs.exists(output)) {
            fs.delete(output, true);
        }

        Job init = Job.getInstance(conf, "PageRank-Init");
        init.setJarByClass(PageRank.class);
        init.setMapperClass(InitMapper.class);
        init.setNumReduceTasks(0);
        init.setOutputKeyClass(Text.class);
        init.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(init, input);
        Path iter0 = new Path(output, "iter0");
        FileOutputFormat.setOutputPath(init, iter0);
        if (!init.waitForCompletion(true)) {
            System.exit(1);
        }

        List<String> pages = new ArrayList<>();
        for (String line : readLines(fs, iter0)) {
            pages.add(line.split("\t")[0]);
        }
        Collections.sort(pages);
        String pageList = String.join(",", pages);
        System.out.println("Total pages: " + pages.size());

        Path prev = iter0;
        for (int i = 1; i <= ITERATIONS; i++) {
            Job job = Job.getInstance(conf, "PageRank-Iteration-" + i);
            job.getConfiguration().set("pagerank.pages", pageList);
            job.setJarByClass(PageRank.class);
            job.setMapperClass(IterMapper.class);
            job.setReducerClass(IterReducer.class);
            job.setNumReduceTasks(1);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);
            FileInputFormat.addInputPath(job, prev);
            Path cur = new Path(output, "iter" + i);
            FileOutputFormat.setOutputPath(job, cur);

            long start = System.currentTimeMillis();
            if (!job.waitForCompletion(true)) {
                System.err.println("Iteration " + i + " failed");
                System.exit(1);
            }
            System.out.printf(Locale.US, "Iteration %d/%d finished, %.1f s%n",
                    i, ITERATIONS, (System.currentTimeMillis() - start) / 1000.0);
            prev = cur;
        }

        Job sort = Job.getInstance(conf, "PageRank-Sort");
        sort.setJarByClass(PageRank.class);
        sort.setMapperClass(SortMapper.class);
        sort.setReducerClass(SortReducer.class);
        sort.setSortComparatorClass(DescDoubleComparator.class);
        sort.setNumReduceTasks(1);
        sort.setMapOutputKeyClass(DoubleWritable.class);
        sort.setMapOutputValueClass(Text.class);
        sort.setOutputKeyClass(Text.class);
        sort.setOutputValueClass(NullWritable.class);
        FileInputFormat.addInputPath(sort, prev);
        Path finalOut = new Path(output, "final");
        FileOutputFormat.setOutputPath(sort, finalOut);
        if (!sort.waitForCompletion(true)) {
            System.exit(1);
        }

        System.out.println();
        System.out.println("===== PageRank result after " + ITERATIONS + " iterations =====");
        for (String line : readLines(fs, finalOut)) {
            System.out.println(line);
        }
    }
}
