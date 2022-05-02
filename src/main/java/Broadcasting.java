import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Broadcasting {

    public static void main(String[] args) {
        testOr();
        System.out.println();
        testAnd();
    }

    public static void testOr() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.constant(Nd4j.createFromArray(true, false, false, true).reshape(2, 2));
        SDVariable b = sd.constant(Nd4j.createFromArray(false, false).reshape(1, 2)); // with .reshape(2) it works
        SDVariable result = sd.math().or(a, b);
        INDArray eval = result.eval();
        System.out.println(eval);
        /*
        expected result:
        [[      true,     false],
         [     false,      true]]

        actual result:
        [[      true,     false],
         [      true,      true]]
        */
    }

    public static void testAnd() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.constant(Nd4j.createFromArray(true, false, false, true).reshape(2, 2));
        SDVariable b = sd.constant(Nd4j.createFromArray(true, true).reshape(1, 2)); // with .reshape(2) it doesn't work either
        SDVariable result = sd.math().and(a, b);
        INDArray eval = result.eval();
        System.out.println(eval);
        /*
        expected result:
        [[      true,     false],
         [     false,      true]]

        actual result:
        [[      true,     false],
         [     false,     false]]
        */
    }
}
