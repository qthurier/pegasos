import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class MapperRegressionPegasos extends Mapper<Object, Text, Text, Text> {
  
 @Override
 public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
	  
	  /* load distributed cache */ 
	  Configuration conf = context.getConfiguration();
	  Path[] cacheFiles = DistributedCache.getLocalCacheFiles(conf);
	  FileInputStream fileStream = new FileInputStream(cacheFiles[0].toString());
	  
	  /* get the percentage of data to keep and epsilon (1.0 are default values) */
	  double pct = conf.getDouble("pct", 1.0);
	  double epsilon = conf.getDouble("epsilon", 1.0);
	  
	  /* read the file storing wt */ 
	  String wString = "";
	  int content;
	  while ((content = fileStream.read()) != -1) wString += (char) content;
	  fileStream.close();
	  
	  /* turn w as a vector */
	  int first_tab, p = conf.getInt("p", 100);
	  String[] weights = wString.split("\t");
	  double[] wArray = new double[p];
	  for(int i=0; i<weights.length; i++) wArray[i] = Double.parseDouble(weights[i]);
	  Vector w = new Vector(wString);
	 
	  /* turn x as a vector and read y */
	  String str = value.toString();
	  first_tab = str.indexOf("\t"); 
	  Vector x = new Vector(str.substring(first_tab + 1));
	  double y = Double.parseDouble(str.substring(0,first_tab));
	  
	  /* emit the absolute difference to reducers */
	  context.write(new Text("mean error"), new Text(String.valueOf(Math.abs((y - w.dot(x))/y))));
	  
	  /* draw cases */ 
	  int nreducers = conf.getInt("nreducers", 2);
	  int nrecords = (int) (int) context.getCounter(DriverRegression.Counters.cases_matching_a_condition).getValue();
	  int redkey = nrecords % nreducers; 
	
	  if(Math.random() <= pct){
		  
		  /* increase the sample size counter (b) */
		  context.getCounter(DriverRegression.Counters.sample_size).increment(1);
		  
		  if(w.dot(x) - y > epsilon){
			  
			  /* increase the counter on the number of cases matching a condition */
			  context.getCounter(DriverRegression.Counters.cases_matching_a_condition).increment(1);
			  
			  /* emit the couple (y, -x) if [<w, x> - y < - epsilon] */
			  Vector xprim = new Vector(x.times(-1.0));
			  Text out = new Text(String.valueOf(y) + '\t' + xprim);
			  context.write(new Text(String.valueOf(redkey)), out);
			  
		  } else if (y - w.dot(x) > epsilon) {
			  
			  /* emit the couple (y, x) if [<w, x> - y > epsilon] */
			  context.write(new Text(String.valueOf(redkey)), value);
			  
		  }
	  }
	  
 }
}
