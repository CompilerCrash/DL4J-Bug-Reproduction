import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionSequenceRecordReader;
import org.datavec.api.writable.IntWritable;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.nd4j.autodiff.listeners.impl.ScoreListener;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.listeners.records.LossCurve;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.weightinit.impl.ZeroInitScheme;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.nd4j.linalg.api.buffer.DataType.FLOAT;

public class Loss {

    public static void main(String[] args) {
        int batchSize = 4;
        int seqLength = 8;

        SameDiff sd = SameDiff.create();

        SDVariable features = sd.squeeze(sd.placeHolder("features", FLOAT, batchSize, 1, seqLength), 1);
        SDVariable labels = sd.squeeze(sd.placeHolder("labels", FLOAT, batchSize, 1, seqLength), 1);
        SDVariable bias = sd.var("bias", new ZeroInitScheme('c'), FLOAT, seqLength);
        SDVariable predictions = features.add("predictions", bias);
        sd.loss.meanSquaredError("loss", labels, predictions, null);

        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Adam(0.1))
                .dataSetFeatureMapping("features")
                .dataSetLabelMapping("labels")
                .build();
        sd.setTrainingConfig(config);

        // Task: add 1 to the inputs
        SequenceRecordReader featureReader = new CollectionSequenceRecordReader(List.of(
                Collections.nCopies(seqLength, Collections.singletonList(new IntWritable(1))),
                Collections.nCopies(seqLength, Collections.singletonList(new IntWritable(2))),
                Collections.nCopies(seqLength, Collections.singletonList(new IntWritable(3))),
                Collections.nCopies(seqLength, Collections.singletonList(new IntWritable(4)))));
        SequenceRecordReader labelReader = new CollectionSequenceRecordReader(List.of(
                Collections.nCopies(seqLength, Collections.singletonList(new IntWritable(2))),
                Collections.nCopies(seqLength, Collections.singletonList(new IntWritable(3))),
                Collections.nCopies(seqLength, Collections.singletonList(new IntWritable(4))),
                Collections.nCopies(seqLength, Collections.singletonList(new IntWritable(5)))));
        DataSetIterator iterator = new SequenceRecordReaderDataSetIterator(featureReader, labelReader, batchSize, 0, true);

        // ScoreListener will consistently report a loss of 0
        History hist = sd.fit(iterator, 10, new ScoreListener(1));

        // The recorded loss values are also constantly 0
        LossCurve curve = hist.lossCurve();
        System.out.println("Loss curve:");
        System.out.println(curve.getLossNames());
        System.out.println(curve.getLossValues());

        // However, the loss calculated here is > 0
        Map<String, INDArray> map = sd.output(iterator, "loss", "bias");
        System.out.println("Final loss: " + map.get("loss"));
        System.out.println("Final bias: " + map.get("bias"));
    }
}
