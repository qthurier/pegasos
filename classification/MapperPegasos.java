import java.io.FileInputStream;
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class MapperPegasos extends Mapper<Object, Text, Text, Text> {
  
 @Override
 public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
	  
	  /* load distributed cache */ 
	  Configuration conf = context.getConfiguration();
	  Path[] cacheFiles = DistributedCache.getLocalCacheFiles(conf);
	  FileInputStream fileStream = new FileInputStream(cacheFiles[0].toString());
	  
	  /* get the percentage of data to keep (second argument is default value) */
	  double pct = conf.getDouble("pct", 1.0);
	  
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
	  
	  /* increase the number of well ranked cases if needed */
	  if(y * w.dot(x) > 0) context.getCounter(Driver.Counters.well_ranked_cases).increment(1);
	  
	  /* draw cases */ 
	  int nreducers = conf.getInt("nreducers", 2);
	  int nrecords = (int) context.getCounter(org.apache.hadoop.mapreduce.TaskCounter.MAP_OUTPUT_RECORDS).getValue();
	  int redkey = nrecords % nreducers; 
	
	  if(Math.random() <= pct){
		  
	   	  /* increase the sample size counter (b) */
		  context.getCounter(Driver.Counters.sample_size).increment(1);	
		  
		  /* emit only if the condition [y * <w, x> < 1] is true */
		  if(y * w.dot(x) < 1) context.write(new Text(String.valueOf(redkey)), value);
		  
	  }
	  
 }
}
