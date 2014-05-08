import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
 
public class ReducerPegasos extends Reducer<Text, Text, NullWritable, Text> {
 
 @Override
 public void reduce(Text key, Iterable<Text> values, Context context)
            throws IOException, InterruptedException {
	 
	 /* get the features dimension, make few definitions/initializations */
	 Configuration conf = context.getConfiguration();
	 int first_tab, p = conf.getInt("p", 100);	 
	 Vector x, gamma_j = new Vector(p);
	 double y;
	 String str;
	 
	 /* loop over all couples (y, x) and increment gamma_j) */
	 for(Text v : values){
		 str = v.toString();
		 first_tab = str.indexOf("\t"); 
		 x = new Vector(str.substring(first_tab + 1));
		 y = Double.parseDouble(str.substring(0,first_tab));
		 gamma_j = gamma_j.plus(x.times(y));
	 } 
	 
	 /* emit gamma_j - there is one gamma_j per reducer - all gamma_j will be summed in the driver in order to build gamma_t */
	 context.write(NullWritable.get(), new Text(gamma_j.toString()));

 }
}