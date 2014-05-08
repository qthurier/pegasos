
import java.net.URI;
import java.nio.charset.Charset;
import java.text.DecimalFormat;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public class DriverRegression {

  /* define job counters */
  static enum Counters {
	  cases_matching_a_condition,
	  sample_size
  }
  
  public static void main(String[] args) throws Exception {
	
	/* register the start time */
	long startTime = System.currentTimeMillis();
    
	/* set up job arguments */
	CommandLine commandLine;
	
	Option option_projection = OptionBuilder.create("projection");
	Option option_reducers = OptionBuilder.isRequired().hasArg().create("reducers"); /* NOTE : we expect at most 10 reducers */
	Option option_p = OptionBuilder.isRequired().hasArg().create("p");
	Option option_T = OptionBuilder.isRequired().hasArg().create("T");
	Option option_b = OptionBuilder.isRequired().hasArg().create("b");
	Option option_epsilon = OptionBuilder.isRequired().hasArg().create("epsilon");
	Option option_lambda = OptionBuilder.isRequired().hasArg().create("lambda");
	Option option_test_set = OptionBuilder.isRequired().hasArg().create("test_set");
	Option option_w_sklearn = OptionBuilder.isRequired().hasArg().create("w_sklearn");
	Option option_w_simulation = OptionBuilder.isRequired().hasArg().create("w_simulation");
    
	Options options = new Options();
    CommandLineParser parser = new GnuParser();
    
    options.addOption(option_projection);
    options.addOption(option_reducers);
    options.addOption(option_p);
    options.addOption(option_T);
    options.addOption(option_b);
    options.addOption(option_epsilon);
    options.addOption(option_lambda);
    options.addOption(option_test_set);
    options.addOption(option_w_sklearn);
    options.addOption(option_w_simulation);
	
	/* job arguments definitions*/	
	boolean project = false;
	double lambda, pct, epsilon;
	int p, T, nReducer;
    String test_set, w_sklearn_loc, w_simulation_loc;
    DecimalFormat df = new DecimalFormat("#####.#####");
    
    try
    {
    	/* parse job arguments */
        commandLine = parser.parse(options, args);
        if (commandLine.hasOption("projection")) project = true;
        T = Integer.parseInt(commandLine.getOptionValue("T"));
		pct = Double.parseDouble(commandLine.getOptionValue("b"));
		epsilon = Double.parseDouble(commandLine.getOptionValue("epsilon"));
		lambda = Double.parseDouble(commandLine.getOptionValue("lambda"));
		nReducer = Integer.parseInt(commandLine.getOptionValue("reducers"));
		p = Integer.parseInt(commandLine.getOptionValue("p"));
		test_set = commandLine.getOptionValue("test_set");
		w_sklearn_loc = commandLine.getOptionValue("w_sklearn");
		w_simulation_loc = commandLine.getOptionValue("w_simulation");
		
		/* print job arguments */
		System.out.println("###### OPTIONS SUMMARY");
		System.out.println("######");
		System.out.println("###### projection = " + project);
		System.out.println("###### number of epochs (T) = " + T);
		System.out.println("###### sample size in percentage = " + pct);
		System.out.println("###### lambda = " + lambda);
		System.out.println("###### epsilon = " + epsilon);
		System.out.println("###### number of reducers = " + nReducer);
		System.out.println("###### features dimension (p) = " + p);
		System.out.println("###### location of the test set on hdfs = " + test_set);
		System.out.println("###### location of the sklearn weights on hdfs = " + w_sklearn_loc);
		System.out.println("###### location of the simulation weights on hdfs = " + w_simulation_loc);
		System.out.println("######");
		
		/* start the job */
		System.out.println("###### START PEGASOS ALGORITHM");
		
		/* 0 - initialize a configuration (the same for all the jobs), read simulation and sklearn weights, make few definitions/initializations*/
	    Configuration conf = new Configuration();
	    conf.setInt("p", p);
	    conf.setInt("nreducers", nReducer);
	    conf.setDouble("pct", pct);   
	    
	    /* open a connexion to hdfs */
	    FileSystem fileSystem = FileSystem.get(conf);
	        
	    /* read simulation weights on hdfs */
	    String w_true_str = "", w_sklearn_str = "";
	    int content; 
	    
	    Path path = new Path(w_simulation_loc);
	    FSDataInputStream in = fileSystem.open(path);
	  	while ((content = in.read()) != -1) w_true_str += (char) content;
	    in.close();
	    
	    /* read sklearn weights on hdfs */
	    path = new Path(w_sklearn_loc);
	    in = fileSystem.open(path);
	  	while ((content = in.read()) != -1) w_sklearn_str += (char) content;
	    in.close();
	    
	    /* close the hdfs connexion */
	    fileSystem.close();
	    
	    /* turn the string to vectors */
	    Vector w_true = new Vector(w_true_str), w_sklearn = new Vector(w_sklearn_str);
	    
	    /* weights initialization */
	    String wString = "0";
	    for(int i=0; i<p-1; i++) wString += "\t" + "0";
		
	    /* define variables which will be used in the loop */
	    Job job;
	    String input, output, sum_error_str;
	    boolean success = false;
		Vector incr1, incr2, w = new Vector(wString);
		byte[] byte_array;
	    int n_going_to_reducers, dataset_size, b, t = 1; 
		double heta, sum_error;
		
	    while(t <= T){
	        
	        /* 1 - write the previous weights on hdfs */
	        fileSystem = FileSystem.get(conf);
			path = new Path("input/pegasos/w.txt");
	        System.out.println("######");
	        System.out.println("###### current iteration = " + t);
	        System.out.println("######");
	        System.out.println("###### write weights on distributed cache");
	        FSDataOutputStream out = fileSystem.create(path);
	        byte_array = w.toString().getBytes(Charset.forName("UTF-8"));
	        out.write(byte_array, 0, byte_array.length);
	        out.close();
	        fileSystem.close();
	  	  
	        /* 2 - copy the previous weights on the distributed cache */
			String pathWithLink= path.toUri().toString() + "#w.txt";
			URI uri= new URI(pathWithLink);
			DistributedCache.addCacheFile(uri, conf);
			DistributedCache.createSymlink(conf);
	        
			/* 3 - global set up of the current job */
			job = Job.getInstance(conf);
	    	job.setMapOutputKeyClass(Text.class);
	        job.setMapOutputValueClass(Text.class);
	    	job.setOutputKeyClass(NullWritable.class);
	        job.setOutputValueClass(Text.class);
	        job.setMapperClass(MapperRegressionPegasos.class); 
	        job.setReducerClass(ReducerRegressionPegasos.class);  
	        job.setInputFormatClass(TextInputFormat.class);
	        job.setOutputFormatClass(TextOutputFormat.class);
	        job.setJarByClass(DriverRegression.class);
	        job.setNumReduceTasks(nReducer);
	        
	        /* set up a dedicated output file for the error sum on the training set */
	        MultipleOutputs.addNamedOutput(job, "error", TextOutputFormat.class, NullWritable.class, Text.class);
	        
	        /* 4 - set up the input/output directories */
	        input = args[0];
	        output = args[1] + t; 
	        FileInputFormat.setInputPaths(job, new Path(input)); 
	        FileOutputFormat.setOutputPath(job, new Path(output)); 
	        
	        /* 5 - submit the current job and wait until it is complete */
	        job.submit();
	        System.out.println("###### run the job");
	        success = job.waitForCompletion(true); 
	        
	        /* 6 - build gamma based on the current job outputs */
	        Vector gamma_j, gamma = new Vector(p);
	        String gamma_str;     
	        fileSystem = FileSystem.get(conf);
	        for(int i=0; i<nReducer; i++){
	            path = new Path(output + "/part-r-0000" + i); 
	            gamma_str = "";
	            in = fileSystem.open(path);
	      	    while ((content = in.read()) != -1) gamma_str += (char) content;
	            in.close();
	            if(gamma_str.equals("")) gamma_j = new Vector(p);
	            else gamma_j = new Vector(gamma_str);
	            gamma = gamma.plus(gamma_j);	      	
	        }
	        fileSystem.close();
	        
	        /* 7 - get the counters values from the current job */
	        b = (int) job.getCounters().findCounter(Counters.sample_size).getValue();
	        n_going_to_reducers = (int) job.getCounters().findCounter(org.apache.hadoop.mapreduce.TaskCounter.MAP_OUTPUT_RECORDS).getValue();
	        dataset_size = (int) job.getCounters().findCounter(org.apache.hadoop.mapreduce.TaskCounter.MAP_INPUT_RECORDS).getValue();
	        
	        /* 8 - print results */
	        
	        /* calculation the error on the test set */   
	        fileSystem = FileSystem.get(conf);
	        
	        path = new Path(test_set); 
	        String str = "";
	        int first_tab, test_set_size = 0;
	        double sum_abs_diff_test_set = 0;
	        in = fileSystem.open(path);
	      	while ((content = in.read()) != -1){
	      		char c = (char) content;
	      		if(c == '\n') {
	      			test_set_size += 1;
	      			first_tab = str.indexOf("\t"); 
	      			Vector x = new Vector(str.substring(first_tab + 1));
	      			double y = Double.parseDouble(str.substring(0, first_tab));
	      			sum_abs_diff_test_set += Math.abs((y - w.dot(x))/y);
	      			str = "";
	      		} else {
	      			str += c;
	      		}
	      	}
	        in.close();
	        
	        /* retrieve the mean error from reducers outputs */
	        path = new Path(output + "/error-r-00000");
	        sum_error_str = "";
	        in = fileSystem.open(path);
	  	    while ((content = in.read()) != -1) sum_error_str += (char) content;
	        in.close();
	        sum_error = Double.parseDouble(sum_error_str);
	        
	        fileSystem.close();
	        
	        System.out.println("###### weights = " + w.toPrettyString());
	        System.out.println("###### mean error to simulation weights = " + df.format((double) w.distanceToL1(w_true)/p));
	        System.out.println("###### mean error to sklearn weights = " + df.format((double) w.distanceToL1(w_sklearn)/p));
	        System.out.println("###### sample size (b) = " + b);
	        System.out.println("###### numbers of cases matching the condition | y - <w, x> | > epsilon  = " + (n_going_to_reducers - dataset_size));
	        System.out.println("###### empirical risk (relative difference) on the training set = " + df.format((double) sum_error/dataset_size));
	        System.out.println("###### empirical risk (relative difference) on the test set = " + df.format((double) sum_abs_diff_test_set/test_set_size));      
	          
	        /* 9 - update the weights */
	        System.out.println("###### update weights");
	        heta = 1/(lambda * ((double) t));
	        incr1 = w.times(1.0 - heta * lambda);       
	        incr2 = gamma.times(heta/((double) b)); 
	        w = incr1.plus(incr2);  
	        
	        /* projection if needed */
	        if(project) {
	        	double m = Math.min(1.0, (1.0/Math.sqrt(lambda))/w.magnitude());
	        	w = w.times(m);
	        }
	        	
	        /* 10 - increment the iteration */
	        t++;  
	   
	    } 
	    
	    /* calculate execution time and print final weights */
	    long stopTime = System.currentTimeMillis();
	    long elapsedTime = (stopTime - startTime)/(1000*60);
	    df = new DecimalFormat("#####.#");
	    System.out.println("######");
	    System.out.println("###### END");
	    System.out.println("######");
	    System.out.println("###### total execution time = " + df.format(elapsedTime) + " min");
	    System.out.println("######");
	    System.out.println("###### final weights = " + w.toPrettyString());
	    
	    System.exit(success ? 0 : 1);

    }
    catch (ParseException exception)
    {
        System.out.print("Job options parsing error: ");
        System.out.println(exception.getMessage());
    } 
    
  }
}

