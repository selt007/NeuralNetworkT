using System;
using System.Threading;

namespace NNT
{
    class Program
    {
        static void Main(string[] args)
        {
            Formula meth;
            meth = new Formula();

            Console.Clear();

            double result = 0, error = 0, ideal = 1;
            //int maxEpoch = 1, trainSet = 4;
            double[] H1 = new double[] { 0.45, -0.12 };
            double[] H2 = new double[] { 0.78, 0.13 };
            double[] O1 = new double[] { 1.5, -2.3 };
            int[] input = new int[] { 1, 0 };
            double Do, Dh1, Dh2, GRADw5, GRADw6;
            
            double h1 = Math.Round(
                meth.Sigmoid(meth.InOut(
                    H1, input)), 2);

            double h2 = Math.Round(
                meth.Sigmoid(meth.InOut(
                    H2, input)), 2);

            double[] arr = new double[] { h1, h2 };

            result = Math.Round(
                meth.Sigmoid(
                    meth.HideOut(arr, O1)),
                2);
            error = Math.Round(
                Math.Pow(
                    (1 - result), 2), 2);

            Do = (1 - error) * ((ideal - error) * error);

            Dh1 = ((1 - h1) * h1) * (O1[0] * Do);
            GRADw5 = h1 * Do;


            Dh2 = ((1 - h2) * h2) * (O1[1] * Do);
            GRADw6 = h2 * Do;

            //double errh11 = (error * H1[0] * h1 * (1 - h1));
            //double errh12 = (error * H1[1] * h1 * (1 - h1));
            //H1[0] = ideal * errh11 * input[0];
            //H1[1] = ideal * errh12 * input[0];

            //double errh21 = (error * H1[0] * h1 * (1 - h1));
            //double errh22 = (error * H1[1] * h1 * (1 - h1));
            //H2[0] = ideal * errh21 * input[1];
            //H2[1] = ideal * errh22 * input[1];

            //O1[0] = ideal * error * h1;
            //O1[1] = ideal * error * h2;

            //Console.WriteLine("Step " + i);
            Console.WriteLine("-----------------");
            Console.WriteLine("Result - " + result + " , |");
            Console.WriteLine("Error - " + error + " .  |");
            Console.WriteLine("----------------- ");

            //Thread.Sleep(1000);
            //Console.Clear();

            //if (maxEpoch > 20) break;
            //else maxEpoch++;

            Console.ReadKey();
        }
    }
}
