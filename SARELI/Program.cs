using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace SARELI
{
    /// <summary>
    /// Class for aligning sequences using different algorithms
    /// </summary>
    internal class Aligner
    {
        private string kernel_file = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().GetName().CodeBase.Substring(8)) + "\\kernel.ptx";

        /// <summary>
        /// Profile with probabilities of each column
        /// </summary>
        public List<Dictionary<char, Double>> profiles;

        /// <summary>
        /// Main sequences abstraction to process
        /// </summary>
        public Sequencer seqToAlign;

        private static Random ran = new Random();

        /// <summary>
        /// Scoring matrix BLOSUM62
        /// Matrix is ordered by the English alphabet to compare, on missing letter is assigned an arbitrary value
        /// example Blosum62 structure:
        ///     A    B   C   D   E   F   G ....
        /// A   4   -2   0  -2  -1  -2   0 ....
        /// B  -2    4  -3   4   1  -3  -1 ....
        /// C   0   -3   9  -3  -4  -2  -3 ....
        /// D  -2    4  -3   6   2  -3  -1 ....
        /// .   .   .   .   .    .   .   . ....
        /// .   .   .   .   .    .   .   . ....
        /// .   .   .   .   .    .   .   . ....
        /// </summary>
        private int[][] Blosum62;

        /// <summary>
        /// Gap Gap scoring value
        /// </summary>
        private int gapgap = -2;

        /// <summary>
        /// Gap mismatch scoring value
        /// </summary>
        private int gapmismatch = -4;

        /// <summary>
        /// Gap opening scoring value
        /// </summary>
        private int gapopening = -5;

        /// <summary>
        /// Constructor for aligner class
        /// </summary>
        /// <param name="seqs">Sequences to work with</param>
        public Aligner(Sequencer seqs)
        {
            seqToAlign = seqs;
            initBlosum62();
        }

        /// <summary>
        /// Blank constructor for aligner
        /// </summary>
        public Aligner()
        {
            seqToAlign = new Sequencer();
            initBlosum62();
        }

        public int[] scoreMatrix = new int[729];

        public void align2cuda_NoRadio(Sequencer seq, int x, int y, int size, double[][] distances, int method = 0)
        {
            int N = seq.count();
            byte[] buff = null;
            int threads = size * size;
            double[] score = new double[seq.count() * seq.count()];

            int[] Sequences_host;
            int[] Sizes_host;

            Stopwatch b = new Stopwatch();
            Stopwatch a = new Stopwatch();
            int sum = 10;

            b.Start();

            using (CudaContext cntxt = new CudaContext(CudaContext.GetMaxGflopsDeviceId()))
            {
                FileStream fs = new FileStream(kernel_file, FileMode.Open, FileAccess.Read);
                BinaryReader br = new BinaryReader(fs);
                long numBytes = new FileInfo(kernel_file).Length;
                buff = br.ReadBytes((int)numBytes);

                CudaDeviceVariable<double> score_d = new CudaDeviceVariable<double>(seq.count() * seq.count());
                CudaDeviceVariable<int> Sizes = new CudaDeviceVariable<int>(seq.count());

                Sizes_host = new int[seq.count()];
                //CudaKernel align2 = cntxt.LoadKernelPTX(buff, getKernelName(kernel_file, "align2SIMO"));
                CudaKernel align2 = cntxt.LoadKernelPTX(buff, getKernelName(kernel_file, "align2SIMO_g"));
                //CudaKernel align2 = cntxt.LoadKernelPTX(buff, getKernelName(kernel_file, "align2SIMO_r"));

                //CudaKernel align2 = cntxt.LoadKernelPTX(buff, getKernelName(kernel_file, "kMerDistance"));

                CudaKernel align2SIMO_Initialize = cntxt.LoadKernelPTX(buff, getKernelName(kernel_file, "align2SIMO_Initialize"));
                align2.BlockDimensions = new dim3(threads);
                align2.GridDimensions = new dim3(1);

                align2SIMO_Initialize.BlockDimensions = new dim3(1);
                align2SIMO_Initialize.GridDimensions = new dim3(1);

                for (int c = 0; c != N; c++)
                {
                    Sizes_host[c] = seq.sequences[c].Count;
                    sum += seq.sequences[c].Count;
                }

                CudaDeviceVariable<int> Sequences = new CudaDeviceVariable<int>(sum);
                Sequences_host = new int[sum];

                int i = 0;
                for (int c = 0; c != seq.count(); c++)
                {
                    for (int d = 0; d != seq.sequences[c].Count; d++)
                    {
                        Sequences_host[i] = seq.sequences[c][d];
                        i++;
                    }
                }

                Sequences.CopyToDevice(Sequences_host);
                Sizes.CopyToDevice(Sizes_host);
                CudaDeviceVariable<int> scoreMatrix_d = new CudaDeviceVariable<int>(729);
                scoreMatrix_d.CopyToDevice(scoreMatrix);
                align2.BlockDimensions = new dim3(threads);
                align2.GridDimensions = new dim3(1);
                CudaDeviceVariable<int> indexes = new CudaDeviceVariable<int>(size * size);

                sum = 0;

                for (int ccc = 0; ccc != size * size; ccc++)
                {
                    if (x * size + (ccc % size) < seq.count() && y * size + (ccc / size) < seq.count())
                    {
                        indexes[ccc] = sum;

                        sum += (Sizes_host[x * size + (ccc % size)] + 1) * (Sizes_host[y * size + (ccc / size)] + 1);
                    }
                }

                int temp = 0;
                temp = cntxt.GetFreeDeviceMemorySize();

                CudaDeviceVariable<int> matrix = new CudaDeviceVariable<int>(sum);
                CudaDeviceVariable<int> matrixDir = new CudaDeviceVariable<int>(sum);
                a.Start();
                //Console.Write("Inicia alineamiento para sacar distancias");
                align2.Run(matrix.DevicePointer, matrixDir.DevicePointer, indexes.DevicePointer, Sequences.DevicePointer, Sizes.DevicePointer, seq.count(), scoreMatrix_d.DevicePointer, x, y, gap(0), gap(1), gap(2), score_d.DevicePointer, size, method);
                // Console.Write("Termina alineamiento para sacar distancias");
                //align2.Run(matrix.DevicePointer, matrixDir.DevicePointer, indexes.DevicePointer, Sequences.DevicePointer, Sizes.DevicePointer, seq.count(), scoreMatrix_d.DevicePointer, x, y, gap(0), gap(1), gap(2), score_d.DevicePointer, size, 4);
                a.Stop();
                //Console.WriteLine("Se tardo en hacer los calculos dentro de la GPU:" + a.ElapsedMilliseconds);

                score_d.CopyToHost(score);

                for (int c = 0; c != size; c++)
                {
                    for (int cc = 0; cc != size; cc++)
                    {
                        if (x * size + cc < N && y * size + c < N)
                        {
                            distances[x * size + cc][y * size + c] = score[((y * size + c) * N) + x * size + cc];
                            distances[y * size + c][x * size + cc] = distances[x * size + cc][y * size + c];
                        }
                    }
                }

                Sequences = null;
                scoreMatrix_d = null;
                Sequences = null;
                GC.Collect();

                b.Stop();
                // Console.WriteLine("Toda la funcion: " + b.ElapsedMilliseconds);
            }
        }

        /// <summary>
        /// method  {column score=0,propossal=1}
        /// </summary>
        /// <param name="seq"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="size"></param>
        /// <param name="distances"></param>
        /// <param name="method"></param>
        public void align2cuda(Sequencer seq, int x, int y, int size, double[][] distances, int method = 0)
        {
            //Console.WriteLine(gap(0) + " " + gap(1) + " " + gap(2));
            int N = seq.count();
            byte[] buff = null;
            int threads = size * size;
            double[] score = new double[seq.count() * seq.count()];

            int[] Sequences_host;
            int[] Sizes_host;

            Stopwatch b = new Stopwatch();
            Stopwatch a = new Stopwatch();
            int sum = 10;

            b.Start();

            using (CudaContext cntxt = new CudaContext(CudaContext.GetMaxGflopsDeviceId()))
            {
                FileStream fs = new FileStream(kernel_file, FileMode.Open, FileAccess.Read);
                BinaryReader br = new BinaryReader(fs);
                long numBytes = new FileInfo(kernel_file).Length;
                buff = br.ReadBytes((int)numBytes);

                CudaDeviceVariable<double> score_d = new CudaDeviceVariable<double>(seq.count() * seq.count());
                CudaDeviceVariable<int> Sizes = new CudaDeviceVariable<int>(seq.count());

                Sizes_host = new int[seq.count()];
                //CudaKernel align2 = cntxt.LoadKernelPTX(buff, getKernelName(kernel_file, "align2SIMO"));
                //CudaKernel align2 = cntxt.LoadKernelPTX(buff, getKernelName(kernel_file, "align2SIMO_g"));

                CudaKernel align2 = cntxt.LoadKernelPTX(buff, getKernelName(kernel_file, "align2SIMO_r"));
                //Console.WriteLine("kf: " + kernel_file + "kname:" + getKernelName(kernel_file, "align2SIMO_r"));
                //CudaKernel align2 = cntxt.LoadKernelPTX(buff, getKernelName(kernel_file, "kMerDistance"));

                CudaKernel align2SIMO_Initialize = cntxt.LoadKernelPTX(buff, getKernelName(kernel_file, "align2SIMO_Initialize"));
                align2.BlockDimensions = new dim3(threads);
                align2.GridDimensions = new dim3(1);

                align2SIMO_Initialize.BlockDimensions = new dim3(1);
                align2SIMO_Initialize.GridDimensions = new dim3(1);

                for (int c = 0; c != N; c++)
                {
                    Sizes_host[c] = seq.sequences[c].Count;
                    sum += seq.sequences[c].Count;
                }

                CudaDeviceVariable<int> Sequences = new CudaDeviceVariable<int>(sum);
                Sequences_host = new int[sum];

                int i = 0;
                for (int c = 0; c != seq.count(); c++)
                {
                    for (int d = 0; d != seq.sequences[c].Count; d++)
                    {
                        Sequences_host[i] = seq.sequences[c][d];
                        i++;
                    }
                }
                for (int xx = 0; xx != 27; xx++)
                {
                    for (int yy = 0; yy != 27; yy++)
                    {
                        scoreMatrix[yy * 27 + xx] = Blosum62[xx][yy];
                    }
                }

                Sequences.CopyToDevice(Sequences_host);
                Sizes.CopyToDevice(Sizes_host);
                CudaDeviceVariable<int> scoreMatrix_d = new CudaDeviceVariable<int>(729);
                scoreMatrix_d.CopyToDevice(scoreMatrix);
                align2.BlockDimensions = new dim3(threads);
                align2.GridDimensions = new dim3(1);
                //Console.WriteLine("(" + x + "," + y + ")/" + (seq.count() / size));
                CudaDeviceVariable<int> indexes = new CudaDeviceVariable<int>(size * size);

                sum = 0;

                for (int ccc = 0; ccc != size * size; ccc++)
                {
                    if (x * size + (ccc % size) < seq.count() && y * size + (ccc / size) < seq.count())
                    {
                        indexes[ccc] = sum;

                        sum += (Sizes_host[x * size + (ccc % size)] + 1) * (Sizes_host[y * size + (ccc / size)] + 1);
                    }
                }

                int temp = 0;
                temp = cntxt.GetFreeDeviceMemorySize();
                //Console.WriteLine("Free memory: " + temp);
                CudaDeviceVariable<int> matrix = new CudaDeviceVariable<int>(sum);
                CudaDeviceVariable<int> matrixDir = new CudaDeviceVariable<int>(sum);
                a.Start();
                //Console.Write("Inicia alineamiento para sacar distancias gap="+gap(0)+" "+gap(1)+" "+ gap(2));
                align2.Run(matrix.DevicePointer, matrixDir.DevicePointer, indexes.DevicePointer, Sequences.DevicePointer, Sizes.DevicePointer, seq.count(), scoreMatrix_d.DevicePointer, x, y, gap(0), gap(1), gap(2), score_d.DevicePointer, size, method);
                // Console.Write("Termina alineamiento para sacar distancias");
                //align2.Run(matrix.DevicePointer, matrixDir.DevicePointer, indexes.DevicePointer, Sequences.DevicePointer, Sizes.DevicePointer, seq.count(), scoreMatrix_d.DevicePointer, x, y, gap(0), gap(1), gap(2), score_d.DevicePointer, size, 4);
                a.Stop();
                //Console.WriteLine("Se tardo en hacer los calculos dentro de la GPU:" + a.ElapsedMilliseconds);

                score_d.CopyToHost(score);

                for (int c = 0; c != size; c++)
                {
                    for (int cc = 0; cc != size; cc++)
                    {
                        if (x * size + cc < N && y * size + c < N)
                        {
                            distances[x * size + cc][y * size + c] = score[((y * size + c) * N) + x * size + cc];
                            distances[y * size + c][x * size + cc] = distances[x * size + cc][y * size + c];
                        }
                    }
                }

                Sequences = null;
                scoreMatrix_d = null;
                Sequences = null;
                GC.Collect();

                b.Stop();
                // Console.WriteLine("Toda la funcion: " + b.ElapsedMilliseconds);
            }
        }

        public double columnScore(Sequencer seq)
        {
            double sum = 0.0;
            Hashtable sym = new Hashtable();

            for (int d = 0; d != seq.sequences.ElementAt(0).Count; d++)
            {
                sym.Clear();
                if (seq.sequences.ElementAt(0).ElementAt(d) != '\n')
                {
                    try
                    {
                        for (int c = 0; c != seq.sequences.Count; c++)
                        {
                            if (sym.ContainsKey(seq.sequences.ElementAt(c).ElementAt(d)))
                            {
                                sym[seq.sequences.ElementAt(c).ElementAt(d)] = (int)sym[seq.sequences.ElementAt(c).ElementAt(d)] + 1;
                            }
                            else
                            {
                                sym.Add(seq.sequences.ElementAt(c).ElementAt(d), 1);
                            }
                        }
                        if (sym.Count == 1)
                        {
                            sum += 1;
                        }
                    }
                    catch (Exception E) {  }
                }
            }
            return sum;
        }

        public Dictionary<char, double>[] profile(Sequencer seq)
        {
            Dictionary<char, double>[] sum = new Dictionary<char, double>[seq.sequences.ElementAt(0).Count];
            double n = 1.0 / (double)seq.count();

            for (int d = 0; d != seq.sequences.ElementAt(0).Count; d++)
            {
                sum[d] = new Dictionary<char, double>();
                for (int c = 0; c != seq.sequences.Count; c++)
                {
                    if (seq.sequences.ElementAt(c).ElementAt(d) != '-' && seq.sequences.ElementAt(c).ElementAt(d) != '\0')
                    {
                        if (sum[d].ContainsKey(seq.sequences.ElementAt(c).ElementAt(d)))
                        {
                            sum[d][seq.sequences.ElementAt(c).ElementAt(d)] = sum[d][seq.sequences.ElementAt(c).ElementAt(d)] + n;
                        }
                        else
                        {
                            sum[d].Add(seq.sequences.ElementAt(c).ElementAt(d), n);
                        }
                    }
                }
            }
            return sum;
        }

        public Sequencer refineAlign3(Sequencer Orig, Sequencer Seqs, double Precision)
        {
            Dictionary<char, double>[] prof = profile(Seqs);
            int n = Seqs.count();
            int[] wrongs = new int[n];
            int L = Seqs.sequences[0].Count;
            bool[] flag = new bool[L];
            int count = 0;
            for (int l = 0; l != wrongs.Length; l++) { wrongs[l] = 0; }
            for (int e = 0; e != prof.Length; e++)
            {
                flag[e] = false;
                for (int d = 0; d != prof[e].Count; d++)
                {
                    if (prof[e].ElementAt(d).Value >= Precision)
                    {
                        flag[e] = true;
                        count++;
                    }
                }
            }
            int[][] locs = new int[count][];
            for (int c = 0; c != count; c++)
            {
                locs[c] = new int[n];
                for (int d = 0; d != n; d++)
                {
                    locs[c][d] = 0;
                }
            }
            int[][] locs2 = new int[count][];
            for (int c = 0; c != count; c++)
            {
                locs2[c] = new int[n];
                for (int d = 0; d != n; d++)
                {
                    locs2[c][d] = 0;
                }
            }

            int last = 0;

            for (int c = 0; c != count; c++)
            {
                for (int d = last; d != L; d++)
                {
                    if (flag[d])
                    {
                        for (int nseq = 0; nseq != n; nseq++)
                        {
                            for (int cc = 0; cc != d; cc++)
                            {
                                if (Seqs.sequences[nseq][cc] != '-')
                                {
                                    locs[c][nseq]++;
                                }
                            }
                        }
                        last = d + 1;
                        break;
                    }
                }
            }

            int test = 0;
            L = Orig.sequences[0].Count;
            for (int c = 0; c != count; c++)
            {
                for (int nseq = 0; nseq != n; nseq++)
                {
                    for (int cc = 0; cc != L; cc++)
                    {
                        if (Orig.sequences[nseq][cc] != '-')
                        {
                            test++;
                        }
                        if (test == locs[c][nseq])
                        {
                            locs2[c][nseq] = test;
                            break;
                        }
                    }
                }
            }
            double average = 0.0;
            for (int c = 0; c != n; c++)
            {
                average += locs[0][c];
            }
            average /= (double)n;
            //Console.WriteLine("Average del primer caracter: " + average);
            //Console.ReadKey();
            return Seqs;
        }

        public Sequencer refineAlign2(Sequencer Seqs, double Precision)
        {
            Dictionary<char, double>[] prof = profile(Seqs);
            int n = Seqs.count();
            int[] wrongs = new int[n];
            int L = Seqs.sequences[0].Count;
            bool[] flag = new bool[L];
            int count = 0;
            for (int l = 0; l != wrongs.Length; l++) { wrongs[l] = 0; }
            for (int e = 0; e != prof.Length; e++)
            {
                flag[e] = false;
                for (int d = 0; d != prof[e].Count; d++)
                {
                    if (prof[e].ElementAt(d).Value >= Precision)
                    {
                        flag[e] = true;
                        count++;
                    }
                }
            }
            //Console.Write("\n" + count + " columnx found that might be realigned\n");
            int co = 0; //Cuenta los epacios entre cada valor true de flags
            int ad = 1; //cuenta en que flag se va
            bool initial = true;
            bool final = false;
            if (count > 1)
            {
                for (int c = 0; c != L - 1; c++)
                {
                    if (flag[c])
                    {
                        ad++;

                        if (co > 3)
                        {
                            if (initial)
                            {
                                for (int d = 0; d != co; d++)
                                {
                                    for (int se = 0; se != n; se++)
                                    {
                                        for (int dd = co - 1; dd != -1; dd--)
                                        {
                                            if (Seqs.sequences.ElementAt(se).ElementAt(dd) != '-' && Seqs.sequences.ElementAt(se).ElementAt(dd + 1) == '-')
                                            {
                                                Seqs.sequences[se][dd + 1] = Seqs.sequences.ElementAt(se).ElementAt(dd);
                                                Seqs.sequences[se][dd] = '-';
                                            }
                                        }
                                    }
                                }
                                initial = false;
                                co = 0;
                            }
                            else
                            {
                                if (final)
                                {
                                    if (c < L - 1)
                                        for (int d = 0; d != L - 1; d++)
                                        {
                                            for (int se = 0; se != n; se++)
                                            {
                                                for (int dd = c + 1; dd != L - 1; dd++)
                                                {
                                                    if (Seqs.sequences.ElementAt(se).ElementAt(dd + 1) != '-' && Seqs.sequences.ElementAt(se).ElementAt(dd) == '-')
                                                    {
                                                        Seqs.sequences[se][dd] = Seqs.sequences.ElementAt(se).ElementAt(dd + 1);
                                                        Seqs.sequences[se][dd + 1] = '-';
                                                    }
                                                }
                                            }
                                        }
                                }
                                else
                                {
                                    for (int d = 0; d != (co / 2); d++)
                                    {
                                        for (int se = 0; se != n; se++)
                                        {
                                            for (int dd = c - 2; dd != c - (co / 2) - 1; dd--)
                                            {
                                                if (Seqs.sequences.ElementAt(se).ElementAt(dd) != '-' && Seqs.sequences.ElementAt(se).ElementAt(dd + 1) == '-')
                                                {
                                                    Seqs.sequences[se][dd + 1] = Seqs.sequences.ElementAt(se).ElementAt(dd);
                                                    Seqs.sequences[se][dd] = '-';
                                                }
                                            }
                                        }
                                    }
                                    for (int d = 0; d != (co / 2); d++)
                                    {
                                        for (int se = 0; se != n; se++)
                                        {
                                            for (int dd = c - co + 1; dd != c - (co / 2); dd++)
                                            {
                                                if (Seqs.sequences.ElementAt(se).ElementAt(dd) == '-' && Seqs.sequences.ElementAt(se).ElementAt(dd + 1) != '-')
                                                {
                                                    Seqs.sequences[se][dd] = Seqs.sequences.ElementAt(se).ElementAt(dd + 1);
                                                    Seqs.sequences[se][dd + 1] = '-';
                                                }
                                            }
                                        }
                                    }
                                }
                                co = -1;
                            }
                        }
                        else
                        {
                            co = -1;
                        }
                    }

                    co++;
                }
            }
            int check;
            for (int c = 0; c != Seqs.sequences[0].Count; c++)
            {
                check = 0;
                for (int cc = 0; cc != Seqs.count(); cc++)
                {
                    if (Seqs.sequences.ElementAt(cc).ElementAt(c) == '-')
                    {
                        check++;
                    }
                }
                if (check == Seqs.count())
                {
                    for (int cc = 0; cc != Seqs.count(); cc++)
                    {
                        Seqs.sequences.ElementAt(cc).RemoveAt(c);
                    }
                    c--;
                }
            }
            return Seqs;
        }

        public Sequencer refineAlign(Sequencer Seqs, double Precision, int iteraciones, bool cuda=true)
        {
            int original = Seqs.count();
            //Console.WriteLine("\nRefining\n");
            bool finiquite = true;
            Sequencer move = new Sequencer();
            Sequencer backup = new Sequencer();

            double initial = new Aligner().columnScore(Seqs);
            int nseq = 1 + (int)Math.Floor((Seqs.count() * (1.0 - Precision)));
            int dd = Seqs.sequences.Count;
            int cols = 0;
            int br = 0;
            for (int gg = 0; finiquite; gg++)
            {
                initial = new Aligner().columnScore(Seqs);
                //Console.Write("\nColumn score:" + initial + "\n");
                backup = new Sequencer();
                for (int c = 0; c != Seqs.count(); c++)
                {
                    backup.addSequence(Seqs.getHeader(c), Seqs.getSequence(c) + "\n");
                }

                if (move.count() > 0)
                {
                    while (move.count() != 0)
                    {
                        move.delete(0);
                    }
                }

                Dictionary<char, double>[] prof = profile(Seqs);
                int[] wrongs = new int[Seqs.count()];

                bool[] flag = new bool[Seqs.sequences[0].Count];
                int count = 0;
                for (int l = 0; l != wrongs.Length; l++) { wrongs[l] = 0; }
                for (int e = 0; e != prof.Length; e++)
                {
                    flag[e] = false;
                    for (int d = 0; d != prof[e].Count; d++)
                    {
                        if (prof[e].ElementAt(d).Value >= Precision)
                        {
                            flag[e] = true;
                            count++;
                        }
                    }
                }
                //Console.Write("\n" + count + " columnx found that might be realigned\n");
                if (gg == 0) { cols = count; }
                if (count == cols) { br++; }
                if (br == iteraciones)
                {
                    finiquite = false;
                    // Console.WriteLine("\nNo enhancement for " + br + " ocasions, aborting");
                    return backup;
                }
                if (count < cols)
                {
                    finiquite = false;
                    //Console.WriteLine("\nNo enhancement in this iteration, aborting");
                    return backup;
                }
                if (count > cols) { br = 0; }

                cols = count;
                if (finiquite)
                {
                    for (int c = 0; c != Seqs.sequences[0].Count; c++)
                    {
                        if (flag[c])
                        {
                            for (int d = 0; d != Seqs.count(); d++)
                            {
                                if (Seqs.sequences.ElementAt(d).ElementAt(c) != '-')
                                    if (prof[c][Seqs.sequences.ElementAt(d).ElementAt(c)] < Precision)
                                    {
                                        wrongs[d]++;
                                    }
                            }
                        }
                    }

                    int pos = 0;

                    int temp = 0;
                    for (int c = pos; c != wrongs.Length; c++)
                    {
                        if (wrongs[c] > wrongs[pos])
                        {
                            temp = wrongs[c];
                            wrongs[c] = wrongs[pos];
                            wrongs[pos] = temp;
                        }
                    }

                    move.addSequence(Seqs.getHeader(pos), Seqs.getSequence(pos).Replace("-", "") + "\n");
                    Seqs.delete(pos);
                    int check = 0;

                    for (int c = 0; c != Seqs.sequences[0].Count; c++)
                    {
                        check = 0;
                        for (int cc = 0; cc != Seqs.count(); cc++)
                        {
                            if (Seqs.sequences.ElementAt(cc).ElementAt(c) == '-')
                            {
                                check++;
                            }
                        }
                        if (check == Seqs.count())
                        {
                            for (int cc = 0; cc != Seqs.count(); cc++)
                            {
                                Seqs.sequences.ElementAt(cc).RemoveAt(c);
                            }
                            c--;
                        }
                    }
                    pos++;
                }

                Aligner refi = new Aligner();
                for (int c = 0; c != move.count(); c++)
                {
                    string s = move.getHeader(c) + "\n" + move.getSequence(c);
                    if (cuda)
                    {
                        Seqs = refi.alignPSPCUDA(Seqs, new Sequencer(s, false),"blosum62");
                    }
                    else {
                        Seqs = refi.alignPSP(Seqs, new Sequencer(s, false),"blosum62");
                    }
                }
            }

            return Seqs;
        }

        public Sequencer refineAlign_s(Sequencer Seqs, double Precision, int iteraciones)
        {
            int original = Seqs.count();
            //Console.WriteLine("\nRefining\n");
            bool finiquite = true;
            Sequencer move = new Sequencer();
            Sequencer backup = new Sequencer();

            double initial = new Aligner().columnScore(Seqs);
            int nseq = 1 + (int)Math.Floor((Seqs.count() * (1.0 - Precision)));
            int dd = Seqs.sequences.Count;
            int cols = 0;
            int br = 0;
            for (int gg = 0; finiquite; gg++)
            {
                initial = new Aligner().columnScore(Seqs);
                //Console.Write("\nColumn score:" + initial + "\n");
                backup = new Sequencer();
                for (int c = 0; c != Seqs.count(); c++)
                {
                    backup.addSequence(Seqs.getHeader(c), Seqs.getSequence(c) + "\n");
                }

                if (move.count() > 0)
                {
                    while (move.count() != 0)
                    {
                        move.delete(0);
                    }
                }

                Dictionary<char, double>[] prof = profile(Seqs);
                int[] wrongs = new int[Seqs.count()];

                bool[] flag = new bool[Seqs.sequences[0].Count];
                int count = 0;
                for (int l = 0; l != wrongs.Length; l++) { wrongs[l] = 0; }
                for (int e = 0; e != prof.Length; e++)
                {
                    flag[e] = false;
                    for (int d = 0; d != prof[e].Count; d++)
                    {
                        if (prof[e].ElementAt(d).Value >= Precision)
                        {
                            flag[e] = true;
                            count++;
                        }
                    }
                }
                //Console.Write("\n" + count + " columnx found that might be realigned\n");
                if (gg == 0) { cols = count; }
                if (count == cols) { br++; }
                if (br == iteraciones)
                {
                    finiquite = false;
                    // Console.WriteLine("\nNo enhancement for " + br + " ocasions, aborting");
                    return backup;
                }
                if (count < cols)
                {
                    finiquite = false;
                    //Console.WriteLine("\nNo enhancement in this iteration, aborting");
                    return backup;
                }
                if (count > cols) { br = 0; }

                cols = count;
                if (finiquite)
                {
                    for (int c = 0; c != Seqs.sequences[0].Count; c++)
                    {
                        if (flag[c])
                        {
                            for (int d = 0; d != Seqs.count(); d++)
                            {
                                if (Seqs.sequences.ElementAt(d).ElementAt(c) != '-')
                                    if (prof[c][Seqs.sequences.ElementAt(d).ElementAt(c)] < Precision)
                                    {
                                        wrongs[d]++;
                                    }
                            }
                        }
                    }

                    int pos = 0;

                    int temp = 0;
                    for (int c = pos; c != wrongs.Length; c++)
                    {
                        if (wrongs[c] > wrongs[pos])
                        {
                            temp = wrongs[c];
                            wrongs[c] = wrongs[pos];
                            wrongs[pos] = temp;
                        }
                    }

                    move.addSequence(Seqs.getHeader(pos), Seqs.getSequence(pos).Replace("-", "") + "\n");
                    Seqs.delete(pos);
                    int check = 0;

                    for (int c = 0; c != Seqs.sequences[0].Count; c++)
                    {
                        check = 0;
                        for (int cc = 0; cc != Seqs.count(); cc++)
                        {
                            if (Seqs.sequences.ElementAt(cc).ElementAt(c) == '-')
                            {
                                check++;
                            }
                        }
                        if (check == Seqs.count())
                        {
                            for (int cc = 0; cc != Seqs.count(); cc++)
                            {
                                Seqs.sequences.ElementAt(cc).RemoveAt(c);
                            }
                            c--;
                        }
                    }
                    pos++;
                }

                Aligner refi = new Aligner();
                for (int c = 0; c != move.count(); c++)
                {
                    string s = move.getHeader(c) + "\n" + move.getSequence(c);
                    Seqs = refi.alignPSPCUDA(Seqs, new Sequencer(s, false));
                }
            }

            return Seqs;
        }

        /// <summary>
        /// Align 2 sequences of the main sequence set using Needleman/Wunsch dynamic programming
        /// Results are saved in the same sequences selected by seqX and seqY parameters
        /// submatrix parameter should be changed on substitution matrix incorporation, and modified the code to adapt to the substitution matrix rules
        /// </summary>
        public void align2(String submatrix = "blosum62", int seqX = 0, int seqY = 1)
        {
            int[][] matrix;
            int[][] matrixDir; //1= (x-1,y-1)  2=(x-1,y)   3=(x,y-1)   -1=(End)

            matrix = new int[seqToAlign.sequences[seqX].Count + 1][];
            matrixDir = new int[seqToAlign.sequences[seqX].Count + 1][];

            for (int c = 0; c != seqToAlign.sequences[seqX].Count + 1; c++)
            {
                matrix[c] = new int[seqToAlign.sequences[seqY].Count + 1];
                matrixDir[c] = new int[seqToAlign.sequences[seqY].Count + 1];
            }
            //Initialize matrix and matrixDir Array

            for (int c = 0; c != matrix.Count(); c++)
            {
                matrix[c][0] = 0;
                matrixDir[c][0] = 2;
            }
            for (int c = 0; c != matrix[0].Count(); c++)
            {
                matrix[0][c] = 0;
                matrixDir[0][c] = 3;
            }

            if (submatrix == "blosum62")
            {
                int topLeft;      //match/mismatch in the diagonal
                int top;      //gap in sequence x
                int left;      //gap in sequence y
                int dir = 0;    //result direction of max[a,b,c]
                int score = 0;  //result of max[a,b,c]
                //  Console.WriteLine(gap(0) + " " + gap(1) + " " + gap(2));
                //Fill the matrix with algorithm computation
                for (int i = 1; i != seqToAlign.sequences[seqX].Count + 1; i++)
                {
                    for (int j = 1; j != seqToAlign.sequences[seqY].Count + 1; j++)
                    {
                        topLeft = matrix[i - 1][j - 1] + Blosum62[seqToAlign.sequences[seqX].ElementAt(i - 1) - 65][seqToAlign.sequences[seqY].ElementAt(j - 1) - 65];

                        int sum = 0;

                        if (matrixDir[i][j - 1] == 1) { sum = gap(1); }
                        top = matrix[i][j - 1] + gap(2) + sum;

                        sum = 0;

                        if (matrixDir[i - 1][j] == 1) { sum = gap(1); }
                        left = matrix[i - 1][j] + gap(2) + sum;

                        if (topLeft >= left && topLeft >= top) { dir = 1; score = topLeft; }
                        if (top > topLeft && top >= left) { dir = 3; score = top; }
                        if (left > topLeft && left > top) { dir = 2; score = left; }

                        matrix[i][j] = score;
                        matrixDir[i][j] = dir;
                    }
                }
                //Do a backtrace for best alignment
                int x = seqToAlign.sequences[seqX].Count;
                int y = seqToAlign.sequences[seqY].Count;
                List<char> sA = new List<char>();
                List<char> sB = new List<char>();

                while (!(x == 0 && y == 0))
                {
                    if (matrixDir[x][y] == 1) { sA.Add(seqToAlign.sequences[seqX][x - 1]); sB.Add(seqToAlign.sequences[seqY][y - 1]); --x; --y; }
                    else
                        if (matrixDir[x][y] == 2) { sA.Add(seqToAlign.sequences[seqX][x - 1]); sB.Add('-'); --x; }
                        else
                            if (matrixDir[x][y] == 3) { sA.Add('-'); sB.Add(seqToAlign.sequences[seqY][y - 1]); --y; }
                }

                int size = sA.Count;
                double g, h;
                double scor = 0;
                for (int c = 0; c != size; c++)
                {
                    for (int d = c - 3; d != c + 3; d++)
                    {
                        if (d > 0 && d < size)
                        {
                            if (d < c)
                            {
                                if (sA[c] != '-' && sB[d] != '-')
                                {
                                    g = (double)Blosum62[(sA[c] - 65)][(sB[d] - 65)];
                                }
                                else
                                {
                                    g = (double)gap(0);
                                }
                                g += 4 - gap(1);
                                h = 3 - (d - (c - 3)) + 1.0;

                                scor += (h / g);
                            }
                            if (d == c)
                            {
                                if (sA[c] != '-' && sB[d] != '-')
                                {
                                    scor += (double)Blosum62[(sA[c] - 65)][(sB[d] - 65)];
                                }
                                else
                                {
                                    scor += gap(0);
                                }
                            }
                            if (d > c)
                            {
                                if (d < size)
                                {
                                    if (sA[c] != '-' && sB[d] != '-')
                                    {
                                        g = (double)Blosum62[(sA[c] - 65)][(sB[d] - 65)];
                                    }
                                    else
                                    {
                                        g = gap(0);
                                    }
                                    g += 4 - gap(1);
                                    h = (d - (c - 3)) + 1.0;
                                    scor += (h / g);
                                }
                            }
                        }
                    }
                }

                seqToAlign.sequences[seqX] = sA.Reverse<char>().ToList<char>();
                seqToAlign.sequences[seqY] = sB.Reverse<char>().ToList<char>();
            }
        }

        public Sequencer alignPSPCUDA(Sequencer a, Sequencer b, String submatrix = "blosum62")
        {
            string result = "";
            string[] aligned;
            int threads = 255;
            int am = a.sequences.Count;
            int an = a.longest();
            int bm = b.sequences.Count;
            int bn = b.longest();
            //Console.WriteLine("an:" + an + "  bn:" + bn);
            int[] SeqA = new int[am * an];
            int[] SeqB = new int[bm * bn];
            int[] inv_Seqs = new int[((am + bm) * (an + bn))];
            int[] alignedSeqs = new int[((am + bm) * (an + bn))];
            int[] matrix = new int[(an + 1) * (bn + 1)];
            int[] matrixDir = new int[(an + 1) * (bn + 1)];
            int[] scoreMatrix = new int[729];
            int[] size = new int[1] { 0 };
            int[] order = new int[an];

            //Transform 2D Scorematrix to vector
            if (submatrix.ToUpper() == "BLOSUM62")
            {
                for (int xx = 0; xx != 27; xx++)
                {
                    for (int yy = 0; yy != 27; yy++)
                    {
                        scoreMatrix[yy * 27 + xx] = Blosum62[xx][yy];
                    }
                }
            }

            if (submatrix.ToUpper() == "SIMPLE")
            {
                for (int x = 0; x != 27; x++)
                {
                    for (int y = 0; y != 27; y++)
                    {
                        if (x != y)
                        {
                            scoreMatrix[y * 27 + x] = -1;
                        }
                        else
                        {
                            scoreMatrix[y * 27 + x] = 2;
                        }
                    }
                }
            }
         //   Console.WriteLine("Gaps: " + gap(0) + " " + gap(1) + " " + gap(2) + " en alignPSPCUDA");
            using (CudaContext cntxt = new CudaContext())
            {
                byte[] buff = null;
                FileStream fs = new FileStream(@kernel_file, FileMode.Open, FileAccess.Read);
                BinaryReader br = new BinaryReader(fs);
                long numBytes = new FileInfo(kernel_file).Length;
                buff = br.ReadBytes((int)numBytes);

                CudaKernel alignPSP = cntxt.LoadKernelPTX(buff, getKernelName(kernel_file, "alignPSP"));

                alignPSP.BlockDimensions = new dim3(threads);
                alignPSP.GridDimensions = new dim3(1);
                alignPSP.DynamicSharedMemory = Convert.ToUInt32(2048);

                CudaKernel tracebackPSP = cntxt.LoadKernelPTX(buff, getKernelName(kernel_file, "tracebackPSP"));
                CudaKernel invertPSP = cntxt.LoadKernelPTX(buff, getKernelName(kernel_file, "invertPSP"));
                tracebackPSP.BlockDimensions = new dim3(1);
                tracebackPSP.GridDimensions = new dim3(1);
                tracebackPSP.DynamicSharedMemory = Convert.ToUInt32(2048);

                invertPSP.BlockDimensions = new dim3(1);
                invertPSP.GridDimensions = new dim3(1);

                CudaDeviceVariable<int> SeqA_d = new CudaDeviceVariable<int>(am * an);
                CudaDeviceVariable<int> sizeAligned = new CudaDeviceVariable<int>(1);
                CudaDeviceVariable<int> SeqB_d = new CudaDeviceVariable<int>(bm * bn);

                CudaDeviceVariable<int> matrix_d = new CudaDeviceVariable<int>((an + 1) * (bn + 1));
                CudaDeviceVariable<int> matrixDir_d = new CudaDeviceVariable<int>((an + 1) * (bn + 1));
                CudaDeviceVariable<int> scoreMatrix_d = new CudaDeviceVariable<int>(729);
                CudaDeviceVariable<int> order_d = new CudaDeviceVariable<int>(an);
                for (int c = 0; c != an; c++)
                {
                    order[c] = 0;
                }
                order_d.CopyToDevice(order);
                CudaDeviceVariable<int> SeqInv_d = new CudaDeviceVariable<int>((am + bm) * (an + bn));
                CudaDeviceVariable<int> alignedSeqs_d = new CudaDeviceVariable<int>((am + bm) * (an + bn));

                scoreMatrix_d.CopyToDevice(scoreMatrix);

                for (int y = 0; y != am; y++)
                {
                    for (int x = 0; x != an; x++)
                    {
                        SeqA[y * an + x] = a.sequences.ElementAt(y).ElementAt(x);
                    }
                }
                for (int y = 0; y != bm; y++)
                {
                    for (int x = 0; x != bn; x++)
                    {
                        SeqB[y * bn + x] = b.sequences.ElementAt(y).ElementAt(x);
                    }
                }
                SeqA_d.CopyToDevice(SeqA);
                SeqB_d.CopyToDevice(SeqB);

                //gg go mis

                int oleada = 0;
                int faltan = an;
                while (faltan > 0)
                {
                    // cntxt.Synchronize();
                    //Console.WriteLine("\nMem: " + cntxt.GetFreeDeviceMemorySize());
                    alignPSP.Run(SeqA_d.DevicePointer, SeqB_d.DevicePointer, matrix_d.DevicePointer, matrixDir_d.DevicePointer, scoreMatrix_d.DevicePointer, am, an, bm, bn, gap(0), gap(1), gap(2), oleada, threads, order_d.DevicePointer);
                    /*order_d.CopyToHost(order);
                   matrixDir_d.CopyToHost(matrixDir);
                   matrix_d.CopyToHost(matrix);

                   for (int a1 =bn-10; a1 != bn+1; a1++)
                   {
                       if (a1 == 0)
                       {
                           Console.Write("     ");
                           for (int a2 = 0; a2 != an; a2++)
                               Console.Write("   " + Convert.ToChar(SeqA[a2]));
                           Console.WriteLine();
                           Console.Write("    ");
                       }
                       for (int a2 = an-10; a2 != an+1; a2++)
                       {
                           if (a1 > 0 && a2 == 0)
                               Console.Write(Convert.ToChar(SeqB[a1 - 1]) + "   ");

                           Console.Write(matrixDir[(an + 1) * a1 + a2] + "   ");
                       }
                       Console.WriteLine();
                   }
                   Console.WriteLine();
                   for (int a1 = bn-10; a1 !=bn+1; a1++)
                   {
                       if (a1 == 0)
                       {
                           Console.Write("     ");
                           for (int a2 = 0; a2 != an; a2++)
                               Console.Write("   " + Convert.ToChar(SeqA[a2]));
                           Console.WriteLine();
                           Console.Write("    ");
                       }
                       for (int a2 = an-10; a2 != an+1; a2++)
                       {
                           if (a1 > 0 && a2 == 0)
                               Console.Write(Convert.ToChar(SeqB[a1 - 1]) + "   ");

                           Console.Write(matrix[(an + 1) * a1 + a2] + "   ");
                       }
                       Console.WriteLine();
                   }*/

                    faltan -= threads;
                    oleada++;
                }

                tracebackPSP.Run(SeqInv_d.DevicePointer, SeqA_d.DevicePointer, SeqB_d.DevicePointer, matrixDir_d.DevicePointer, am, an, bm, bn, sizeAligned.DevicePointer);
                SeqInv_d.CopyToHost(inv_Seqs);
                sizeAligned.CopyToHost(size);

                invertPSP.Run(alignedSeqs_d.DevicePointer, SeqInv_d.DevicePointer, (am + bm), sizeAligned.DevicePointer, (an + bn));

                alignedSeqs_d.CopyToHost(alignedSeqs);
                char[] s = Array.ConvertAll(alignedSeqs, x => (char)x);

                char[] k = new char[size[0]];
                for (int d = 0; d != am + bm; d++)
                {
                    if (d < am) { result += a.getHeader(d) + "\n"; } else { result += b.getHeader(d - am) + "\n"; }

                    Array.Copy(s, (an + bn) * d, k, 0, size[0]);
                    result += (new string(k));

                    result += "\n";
                }
                /*using (System.IO.StreamWriter file = new System.IO.StreamWriter("debug.txt", true))
                {
                    file.WriteLine(result);
                }*/
                SeqA_d = null;
                SeqA_d = null;
                sizeAligned = null;
                SeqB_d = null;
                matrix_d = null;
                matrixDir_d = null;
                scoreMatrix_d = null;
                SeqInv_d = null;
                alignedSeqs_d = null;
                matrix = null;
                matrixDir = null;
                SeqA = null;
                SeqB = null;
                GC.Collect();
            }

            aligned = result.Split('|');

            Sequencer res = new Sequencer(result, false);
      //      res.print(0, -1, true);
            return res;
        }

        /// <summary>
        /// Align 2 set of sequences ([A] and [B]), using Profile Sum of Pairs algorithm, and [submatrix] substitution matrix.
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <param name="submatrix"></param>
        /// <returns></returns>
        public Sequencer alignPSP(Sequencer A, Sequencer B, String submatrix = "SIMPLE")
        {
//Console.WriteLine("Gaps: " + gap(0) + " " + gap(1) + " " + gap(2)+" en alignPSP");
            int [][]ScoreMatrix=new int[27][];
            for (int c = 0; c != 27; c++)
            {
                ScoreMatrix[c] = new int[27];
            }
            if (submatrix.ToUpper()== "BLOSUM62") {
                for (int x = 0; x != 27; x++) {
                    for (int y = 0; y != 27; y++) {
                        ScoreMatrix[x][y] = Blosum62[x][y];
                    }
                }
            }

            if (submatrix.ToUpper() == "SIMPLE")
            {
                for (int x = 0; x != 27; x++)
                {
                    for (int y = 0; y != 27; y++)
                    {
                        if (x == y)
                        {
                            ScoreMatrix[x][y] = 2;
                        }
                        else {
                            ScoreMatrix[x][y] = -1;
                        }
                    }
                }
            }

            Sequencer aligned = new Sequencer();
            int ka = A.longest() + 1;
            int kb = B.longest() + 1;
            int na = A.count();
            int nb = B.count();

            int[][] matrix = new int[ka][];
            int[][] matrixDir = new int[ka][];

            for (int c = 0; c != ka; c++)
            {
                matrix[c] = new int[kb];
                matrixDir[c] = new int[kb];
            }
            matrix[0][0] = 0;
            matrixDir[0][0] = 0;

            for (int c = 1; c != ka; c++)
            {
                matrix[c][0] = gap(2) * c;
                matrixDir[c][0] = 2;
            }
            for (int c = 1; c != kb; c++)
            {
                matrix[0][c] = gap(2) * c;
                matrixDir[0][c] = 3;
            }
            for (int x = 1; x != ka; x++)
            {
                for (int y = 1; y != kb; y++)
                {
                    int temp1 = A.count();
                    int temp2 = B.count();
                    int g0 = gap(0);
                    int g2 = gap(2);
                    
                      int sum = 0;
                      for (int xx = 0; xx != temp1; xx++)
                      {
                          for (int yy = 0; yy != temp2; yy++)
                        {
                            sum += A.sequences[xx][x - 1] != '-' && B.sequences[yy][y - 1] != '-' ? ScoreMatrix[A.sequences[xx][x - 1] - 65][B.sequences[yy][y - 1] - 65] : A.sequences[xx][x - 1] == B.sequences[yy][y - 1] ? gap(0) : gap(2);
                        }
                      }
                    int tl = matrix[x - 1][y - 1] +  sum;;
                    sum = 0;
                    for (int xx = 0; xx != temp1; xx++)
                    {
                        for (int yy = 0; yy != temp2; yy++)
                        {
                            sum += A.sequences[xx][x-1] == '-' ? g0 : g2;
                        }
                    }

                    int t = matrix[x][y - 1] + sum;
                    sum = 0;

                    for (int xx = 0; xx != temp1; xx++)
                    {
                        for (int yy = 0; yy != temp2; yy++)
                        {
                            sum += B.sequences[yy][y-1] == '-' ? g0 : g2;
                        }
                    }
                    int l = matrix[x - 1][y] + sum;

                    /*
                    int tl = matrix[x - 1][y - 1] + sumOfPairs(A, B, x - 1, y - 1, "simple");
                    int t = matrix[x][y - 1] + sumOfPairs(A, B, -1, y - 1, "simple") ;
                    int l = matrix[x - 1][y] + sumOfPairs(A, B, x - 1, -1, "simple");
                    */
                    if (tl >= t && tl >= l) { matrix[x][y] = tl; matrixDir[x][y] = 1; }
                    if (t > tl && t >= l) { matrix[x][y] = t; matrixDir[x][y] = 3; }
                    if (l > t && l > tl) { matrix[x][y] = l; matrixDir[x][y] = 2; }
                }
            }
            /*
            for (int y = 0; y != kb; y++)
            {
                for (int x = 0; x != ka; x++)
                {
                    Console.Write(matrixDir[x][y] + " ");
                }
                Console.WriteLine();
            }*/

            int j = kb - 1, i = ka - 1;

            while (!(i == 0 && j == 0))
            {
                if (matrixDir[i][j] == 3)
                {
                    //for (int c = 0; c != na; c++) { A.sequences[c].Insert(ka - 1 - j, '-'); }
                    for (int c = 0; c != na; c++) { A.sequences[c].Insert(i, '-'); }

                    j--;
                }
                if (matrixDir[i][j] == 2)
                {
                    //for (int c = 0; c != nb; c++) { B.sequences[c].Insert(kb - 1 - i, '-'); }
                    for (int c = 0; c != nb; c++) { B.sequences[c].Insert(j, '-'); }

                    i--;
                }
                if (matrixDir[i][j] == 1)
                {
                    j--;
                    i--;
                }
            }

            for (int c = 0; c != na; c++) { aligned.addSequence(A.getHeader(c), A.getSequence(c)); }

            for (int c = 0; c != nb; c++) { aligned.addSequence(B.getHeader(c), B.getSequence(c)); }
           // aligned.print(0, -1, true);
            return aligned;
        }

        public double[] averageDistance(Sequencer all, string distanceMeasure = "propossal", int ra = 5, int size = 5)
        {
            double[] res = new double[2];
            Sequencer[] allDivided;
            Sequencer[] aligning;

            Aligner worker2 = new Aligner(all);
            allDivided = new Sequencer[all.count()];
            all.divide(allDivided);
            int N = worker2.seqToAlign.count();
            aligning = new Sequencer[N];

            double[][] distances = new double[N + 1][];
            int[] headers = new int[N + 1];
            double[][] qmatrix = new double[N + 1][];

            for (int c = 0; c != N + 1; c++)
            {
                distances[c] = new double[N + 1];
                qmatrix[c] = new double[N + 1];
            }
            // Console.Write("\nMeasuring sequences\n");
            //Aligner scorer;
            worker2.gap(-2, -4, -3);
            int kk = (int)Math.Ceiling(((double)N / (double)size));
            if (N > size)
            {
                for (int c = 0; c != kk; c++)
                {
                    for (int cc = c; cc != kk; cc++)
                    {
                        //Console.Write("\r                                               \rDistancias: " + c + "/" + kk + "  " + cc + "/" + kk);
                        worker2.align2cuda_NoRadio(all, c, cc, size, distances, ra);
                    }
                }
            }
            else
            {
                worker2.align2cuda_NoRadio(all, 0, 0, size, distances, ra);
            }
            double e = 0;
            res[0] = 0;
            for (int c = 0; c != distances.Length; c++)
            {
                for (int d = c; d != distances.Length; d++)
                {
                    if (c != d)
                    {
                        res[0] += distances[c][d];
                        e++;
                    }
                }
            }
            res[0] = res[0] / e;
            res[1] = 0;
            for (int c = 0; c != distances.Length; c++)
            {
                for (int d = c; d != distances.Length; d++)
                {
                    if (c != d)
                    {
                        res[1] += Math.Pow(distances[c][d] - res[0], 2.0);
                    }
                }
            }
            res[1] = Math.Sqrt(res[1] / e);

            return res;
        }

        /// <summary>
        /// This function align an arbitrary sequence of type Sequencer and return the aligned sequences in the same format as the input
        /// The algorithm used in this function is the Profile Sum of Pairs with the NeighbourJoining for ordering.
        /// </summary>
        /// <param name="all">All sequences to align</param>
        /// <param name="distanceMeasure">Distance measure to be used (propossal,simple,feng,sum)</param>
        /// <param name="ra">Parameter of the propossal distance: Radius</param>
        /// <returns></returns>
        public Sequencer alignByNeighbourJoiningCUDA(Sequencer all, string distanceMeasure = "propossal", int ra = 5, int size = 5)
        {
            Sequencer[] allDivided;
            Sequencer[] aligning;

            Aligner worker2 = new Aligner(all);
            allDivided = new Sequencer[all.count()];
            all.divide(allDivided);
            int N = worker2.seqToAlign.count();
            aligning = new Sequencer[N];

            double[][] distances = new double[N + 1][];
            int[] headers = new int[N + 1];
            double[][] qmatrix = new double[N + 1][];

            for (int c = 0; c != N + 1; c++)
            {
                distances[c] = new double[N + 1];
                qmatrix[c] = new double[N + 1];
            }
            // Console.Write("\nMeasuring sequences\n");
            //Aligner scorer;
            int kk = (int)Math.Ceiling(((double)N / (double)size));
            if (N > size)
            {
                for (int c = 0; c != kk; c++)
                {
                    for (int cc = c; cc != kk; cc++)
                    {
                        worker2.align2cuda(all, c, cc, size, distances, ra);
                    }
                }
            }
            else
            {
                worker2.align2cuda(all, 0, 0, size, distances, ra);
            }
            for (int x = 0; x != N; x++)
            {
                headers[x] = x;
            }
            /*
            for (int x = 0; x != N; x++)
            {
                headers[x] = x;
                for (int y = x; y != N; y++)
                {
                    //Console.Write("Distancia ["+x+"] vs ["+y+"] de "+N+"\n");
                    if (x != y)
                    {
                        if (distanceMeasure == "propossal")
                            distances[x][y] = worker2.distancePropossal(x, y, ra);
                        if (distanceMeasure == "simple")
                            distances[x][y] = worker2.simpleDistance(x, y);
                        if (distanceMeasure == "feng")
                            distances[x][y] = worker2.distanceFengDoolittle2(x, y);
                        if (distanceMeasure == "sum")
                            distances[x][y] = worker2.SumOfPairs2(x, y) / (worker2.SumOfPairs2(x, x) + worker2.SumOfPairs2(y, y) / 2.0);

                        distances[y][x] = distances[x][y];
                    }
                }
            }
          */
            headers[N] = N;
            /*
            Console.WriteLine("First, Distances:");
            for (int x = 0; x != N; x++)
            {
                for (int y = 0; y != N; y++)
                {
                    Console.Write(distances[y][x] + "\t");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
            */
            //Console.Write("\nAligning sequences\n");
            int k = N + (worker2.seqToAlign.count() - N);
            int T = N;  //Offset to detect if the tree node is a subnode or original node (Above N, t is a subnode)
            int NN = N; //Number of original nodes
            while (N > 1)
            {
                //Console.WriteLine(N + " de " + NN);
                double smallest = 1000000;
                int indexSmallestX = -1;
                int indexSmallestY = -1;
                for (int x = 0; x != N; x++)
                {
                    for (int y = 0; y != N; y++)
                    {
                        if (x != y)
                        {
                            double sumx = 0.0;
                            double sumy = 0.0;
                            for (int c = 0; c != N; c++)
                            {
                                sumx += distances[x][c];
                                sumy += distances[y][c];
                            }
                            qmatrix[x][y] = (N - 2) * distances[x][y] - (sumx + sumy);
                            if (qmatrix[x][y] <= smallest)
                            {
                                smallest = qmatrix[x][y];
                                indexSmallestX = x;
                                indexSmallestY = y;
                            }
                        }
                    }
                }
                //Console.WriteLine("merge: " + headers[indexSmallestX] + " and " + headers[indexSmallestY]);
                if (headers[indexSmallestX] >= NN)
                {
                    if (headers[indexSmallestY] >= NN)
                    {
                        aligning[T - NN] = worker2.alignPSPCUDA(aligning[headers[indexSmallestX] - NN], aligning[headers[indexSmallestY] - NN], "simple");
                    }
                    else
                    {
                        aligning[T - NN] = worker2.alignPSPCUDA(aligning[headers[indexSmallestX] - NN], allDivided[headers[indexSmallestY]], "simple");
                    }
                }
                else
                {
                    if (headers[indexSmallestY] >= NN)
                    {
                        aligning[T - NN] = worker2.alignPSPCUDA(allDivided[headers[indexSmallestX]], aligning[headers[indexSmallestY] - NN], "simple");
                    }
                    else
                    {
                        aligning[T - NN] = worker2.alignPSPCUDA(allDivided[headers[indexSmallestX]], allDivided[headers[indexSmallestY]], "simple");
                    }
                }
                T++;
                double sumsmx = 0;
                double sumsmy = 0;
                double di = 0.0;
                for (int c = 0; c != N; c++)
                {
                    sumsmx += distances[indexSmallestX][c];
                    sumsmy += distances[indexSmallestY][c];
                }
                di = .5 * distances[indexSmallestX][indexSmallestY] + ((1.0 / (2.0 * (N - 2.0))) * (Double)(sumsmx - sumsmy));
                for (int c = 0; c != N + 1; c++)
                {
                    if (c != N)
                    {
                        if (c == indexSmallestX) { distances[N][c] = di; distances[c][N] = di; }
                        if (c == indexSmallestY) { distances[N][c] = distances[indexSmallestX][indexSmallestY] - di; distances[c][N] = distances[indexSmallestX][indexSmallestY] - di; }
                        if (c != indexSmallestX && c != indexSmallestY) { distances[N][c] = (.5 * distances[indexSmallestX][indexSmallestY]) + (1 / (2 * (N - 1))) * (sumsmx - sumsmy); distances[c][N] = distances[N][c]; }
                    }
                }
                indexSmallestY = (int)headers[indexSmallestY];
                for (int c = 0; c != N + 1; c++)
                {
                    double t = 0.0;
                    t = distances[0][c];
                    distances[0][c] = distances[indexSmallestX][c];
                    distances[indexSmallestX][c] = t;
                    t = distances[c][0];
                    distances[c][0] = distances[c][indexSmallestX];
                    distances[c][indexSmallestX] = t;
                }
                int te = 0;
                te = headers[0];
                headers[0] = headers[indexSmallestX];
                headers[indexSmallestX] = te;
                for (int c = 0; c != N + 1; c++)
                {
                    if (headers[c] == indexSmallestY) { indexSmallestY = c; break; }
                }
                for (int c = 0; c != N + 1; c++)
                {
                    double t = 0.0;
                    t = distances[1][c];
                    distances[1][c] = distances[indexSmallestY][c];
                    distances[indexSmallestY][c] = t;
                    t = distances[c][1];
                    distances[c][1] = distances[c][indexSmallestY];
                    distances[c][indexSmallestY] = t;
                }
                te = headers[1];
                headers[1] = headers[indexSmallestY];
                headers[indexSmallestY] = te;
                for (int x = 2; x != N + 1; x++)
                {
                    headers[x - 2] = headers[x];
                }
                ++k;
                headers[N - 1] = k;
                for (int x = 2; x != N + 1; x++)
                    for (int y = 2; y != N + 1; y++)
                    {
                        distances[x - 2][y - 2] = distances[x][y];
                    }
                N--;
                //Console.Write("\r                 \rLeft:" + N + "   ");
            }
            //Console.Write("\n");
            return aligning[NN - 2];
        }

        /// <summary>
        /// This function align an arbitrary sequence of type Sequencer and return the aligned sequences in the same format as the input
        /// The algorithm used in this function is the Profile Sum of Pairs with the NeighbourJoining for ordering.
        /// </summary>
        /// <param name="all">All sequences to align</param>
        /// <param name="distanceMeasure">Distance measure to be used (propossal,simple,feng,sum)</param>
        /// <param name="ra">Parameter of the propossal distance: Radius</param>
        /// <returns></returns>
        public Sequencer alignByUPGMACUDA(Sequencer all, string distanceMeasure = "propossal", int ra = 5, int size = 5)
        {
            Sequencer[] allDivided;
            Sequencer[] aligning;

            Aligner worker2 = new Aligner(all);
            allDivided = new Sequencer[all.count()];
            all.divide(allDivided);
            int N = worker2.seqToAlign.count();
            aligning = new Sequencer[N];

            double[][] distances = new double[N + 1][];
            int[] headers = new int[N + 1];
            double[][] qmatrix = new double[N + 1][];

            for (int c = 0; c != N + 1; c++)
            {
                distances[c] = new double[N + 1];
                qmatrix[c] = new double[N + 1];
            }

            //Aligner scorer;

            if (N > size)
            {
                for (int c = 0; c != (int)Math.Ceiling(((double)N / (double)size)); c++)
                {
                    for (int cc = c; cc != (int)Math.Ceiling(((double)N / (double)size)); cc++)
                    {
                        worker2.align2cuda(all, c, cc, size, distances, 0);
                    }
                }
            }
            else
            {
                worker2.align2cuda(all, 0, 0, size, distances, 0);
            }
            for (int x = 0; x != N; x++)
            {
                headers[x] = x;
            }
            /*
            for (int x = 0; x != N; x++)
            {
                headers[x] = x;
                for (int y = x; y != N; y++)
                {
                    //Console.Write("Distancia ["+x+"] vs ["+y+"] de "+N+"\n");
                    if (x != y)
                    {
                        if (distanceMeasure == "propossal")
                            distances[x][y] = worker2.distancePropossal(x, y, ra);
                        if (distanceMeasure == "simple")
                            distances[x][y] = worker2.simpleDistance(x, y);
                        if (distanceMeasure == "feng")
                            distances[x][y] = worker2.distanceFengDoolittle2(x, y);
                        if (distanceMeasure == "sum")
                            distances[x][y] = worker2.SumOfPairs2(x, y) / (worker2.SumOfPairs2(x, x) + worker2.SumOfPairs2(y, y) / 2.0);

                        distances[y][x] = distances[x][y];
                    }
                }
            }
            */
            headers[N] = N;

            /*Console.WriteLine("First, Distances:");
            for (int x = 0; x != N; x++)
            {
                for (int y = 0; y != N; y++)
                {
                    Console.Write(distances[y][x] + "\t");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
            */
            int k = N + (worker2.seqToAlign.count() - N);
            int T = N;  //Offset to detect if the tree node is a subnode or original node (Above N, t is a subnode)
            int NN = N; //Number of original nodes
            while (N > 1)
            {
                //Console.WriteLine(N + " de " + NN);
                double smallest = 1000000;
                int indexSmallestX = -1;
                int indexSmallestY = -1;
                for (int x = 0; x != N; x++)
                {
                    for (int y = x; y != N; y++)
                    {
                        if (x != y)
                        {
                            if (distances[x][y] < smallest)
                            {
                                indexSmallestX = x;
                                indexSmallestY = y;
                                smallest = distances[x][y];
                            }
                        }
                    }
                }
                if (headers[indexSmallestX] >= NN)
                {
                    if (headers[indexSmallestY] >= NN)
                    {
                        aligning[T - NN] = worker2.alignPSPCUDA(aligning[headers[indexSmallestX] - NN], aligning[headers[indexSmallestY] - NN], "simple");
                    }
                    else
                    {
                        aligning[T - NN] = worker2.alignPSPCUDA(aligning[headers[indexSmallestX] - NN], allDivided[headers[indexSmallestY]], "simple");
                    }
                }
                else
                {
                    if (headers[indexSmallestY] >= NN)
                    {
                        aligning[T - NN] = worker2.alignPSPCUDA(allDivided[headers[indexSmallestX]], aligning[headers[indexSmallestY] - NN], "simple");
                    }
                    else
                    {
                        aligning[T - NN] = worker2.alignPSPCUDA(allDivided[headers[indexSmallestX]], allDivided[headers[indexSmallestY]], "simple");
                    }
                }
                T++;

                for (int c = 0; c != N; c++)
                {
                    if (c != indexSmallestX && c != indexSmallestY)
                    {
                        distances[N][c] = (distances[c][indexSmallestX] + distances[c][indexSmallestY]) / (T - NN + 2);
                        distances[c][N] = distances[N][c];
                    }
                }
                indexSmallestY = (int)headers[indexSmallestY];
                for (int c = 0; c != N + 1; c++)
                {
                    double t = 0.0;
                    t = distances[0][c];
                    distances[0][c] = distances[indexSmallestX][c];
                    distances[indexSmallestX][c] = t;
                    t = distances[c][0];
                    distances[c][0] = distances[c][indexSmallestX];
                    distances[c][indexSmallestX] = t;
                }
                int te = 0;
                te = headers[0];
                headers[0] = headers[indexSmallestX];
                headers[indexSmallestX] = te;
                for (int c = 0; c != N + 1; c++)
                {
                    if (headers[c] == indexSmallestY) { indexSmallestY = c; break; }
                }
                for (int c = 0; c != N + 1; c++)
                {
                    double t = 0.0;
                    t = distances[1][c];
                    distances[1][c] = distances[indexSmallestY][c];
                    distances[indexSmallestY][c] = t;
                    t = distances[c][1];
                    distances[c][1] = distances[c][indexSmallestY];
                    distances[c][indexSmallestY] = t;
                }
                te = headers[1];
                headers[1] = headers[indexSmallestY];
                headers[indexSmallestY] = te;
                for (int x = 2; x != N + 1; x++)
                {
                    headers[x - 2] = headers[x];
                }
                ++k;
                headers[N - 1] = k;
                for (int x = 2; x != N + 1; x++)
                    for (int y = 2; y != N + 1; y++)
                    {
                        distances[x - 2][y - 2] = distances[x][y];
                    }
                N--;
            }
            return aligning[NN - 2];
        }

        /// <summary>
        /// Align 2 sequences of the main sequence set (seqToAlign[seqX] and seqToAlign[seqY]) using Needleman/Wunsch dynamic programming and CUDA technology
        /// This function uses kernels from other project added at the end of this file "align2", "invert" and "traceback".
        /// </summary>
        /// <param name="submatrix"></param>
        /// <param name="seqX"></param>
        /// <param name="seqY"></param>
        public void cudaAlign2(String submatrix = "blosum62", int seqX = 0, int seqY = 1)
        {
            int m = (seqToAlign.sequences[seqX].Count + 1);
            int n = (seqToAlign.sequences[seqY].Count + 1);
            int[] SeqA = new int[m + n - 2];
            int[] SeqB = new int[m + n - 2];
            int[] matrix = new int[m * n];
            int[] matrixDir = new int[m * n];
            int[] scoreMatrix = new int[729];
            int[] size = new int[1] { 0 };
            //Transform 2D Scorematrix to vector
            for (int x = 0; x != 27; x++)
            {
                for (int y = 0; y != 27; y++)
                {
                    scoreMatrix[y * 27 + x] = Blosum62[x][y];
                }
            }

            //Stopwatch timer = new Stopwatch();

            using (CudaContext cntxt = new CudaContext())
            {
                CUmodule moduleAlign2 = cntxt.LoadModule(kernel_file);
                CudaKernel align2 = new CudaKernel(getKernelName(kernel_file, "kernel"), moduleAlign2, cntxt) { GridDimensions = new dim3(1, 1), BlockDimensions = new dim3(m, 1) };

                CudaKernel traceback = new CudaKernel(getKernelName(kernel_file, "traceback"), moduleAlign2, cntxt) { GridDimensions = new dim3(1, 1), BlockDimensions = new dim3(1, 1) };

                CudaDeviceVariable<int> SeqA_d = new CudaDeviceVariable<int>(m + n - 2);
                CudaDeviceVariable<int> sizeAligned = new CudaDeviceVariable<int>(1);
                CudaDeviceVariable<int> SeqB_d = new CudaDeviceVariable<int>(m + n - 2);
                CudaDeviceVariable<int> SeqAinv_d = new CudaDeviceVariable<int>(m + n - 2);
                CudaDeviceVariable<int> SeqBinv_d = new CudaDeviceVariable<int>(m + n - 2);
                CudaDeviceVariable<int> matrix_d = new CudaDeviceVariable<int>(m * n);
                CudaDeviceVariable<int> matrixDir_d = new CudaDeviceVariable<int>(m * n);
                CudaDeviceVariable<int> scoreMatrix_d = new CudaDeviceVariable<int>(729);
                align2.DynamicSharedMemory = Convert.ToUInt32(m * 4);
                scoreMatrix_d.CopyToDevice(scoreMatrix);
                for (int x = 0; x != m - 1; x++)
                {
                    SeqA[x] = seqToAlign.sequences.ElementAt(0).ElementAt(x);
                }
                for (int x = 0; x != n - 1; x++)
                {
                    SeqB[x] = seqToAlign.sequences.ElementAt(1).ElementAt(x);
                }
                SeqA_d.CopyToDevice(SeqA);
                SeqB_d.CopyToDevice(SeqB);
                align2.Run(SeqA_d.DevicePointer, SeqB_d.DevicePointer, matrix_d.DevicePointer, matrixDir_d.DevicePointer, scoreMatrix_d.DevicePointer, m, n);
                matrixDir_d.CopyToHost(matrixDir);
                traceback.Run(SeqA_d.DevicePointer, SeqAinv_d.DevicePointer, SeqB_d.DevicePointer, SeqBinv_d.DevicePointer, matrixDir_d.DevicePointer, m, n, sizeAligned.DevicePointer);
                sizeAligned.CopyToHost(size);
                CudaKernel invert = new CudaKernel(getKernelName(kernel_file, "invert"), moduleAlign2, cntxt) { GridDimensions = new dim3(1, 1), BlockDimensions = new dim3(size[0], 1) };
                invert.Run(SeqA_d.DevicePointer, SeqAinv_d.DevicePointer, SeqB_d.DevicePointer, SeqBinv_d.DevicePointer, sizeAligned.DevicePointer);
                SeqA_d.CopyToHost(SeqA);
                SeqB_d.CopyToHost(SeqB);
            }
        }

        /// <summary>
        /// This function align an arbitrary sequence of type Sequencer and return the aligned sequences in the same format as the input
        /// The algorithm used in this function is the Profile Sum of Pairs with the NeighbourJoining for ordering, the distance function
        /// used is the simpleDistance
        /// </summary>
        /// <param name="all">Sequences to be aligned</param>
        /// <returns></returns>
        public Sequencer alignByNeighbourJoining(Sequencer all, int ra = 3)
        {
            Sequencer[] allDivided;
            Sequencer[] aligning;
            Aligner worker2 = new Aligner(all);
            allDivided = new Sequencer[all.count()];
            all.divide(allDivided);
            int N = worker2.seqToAlign.count();
            aligning = new Sequencer[N];
            double[][] distances = new double[N + 1][];
            int[] headers = new int[N + 1];
            double[][] qmatrix = new double[N + 1][];
            for (int c = 0; c != N + 1; c++)
            {
                distances[c] = new double[N + 1];
                qmatrix[c] = new double[N + 1];
            }
            for (int x = 0; x != N; x++)
            {
                headers[x] = x;
                for (int y = x; y != N; y++)
                {
                    if (x != y)
                    {
                        //distances[x][y] = worker2.simpleDistance(x, y);
                        distances[x][y] = worker2.distancePropossal(x, y, ra);

                        //distances[x][y] = worker2.FengDoolittle2(x, y);
                        //distances[x][y] =  (worker2.SumOfPairs2(x, x)+ worker2.SumOfPairs2(y, y)/2.0)+worker2.SumOfPairs2(x, y);
                        distances[y][x] = distances[x][y];
                    }
                }
            }
            headers[N] = N;

            /*Console.WriteLine("First, Distances:");
            for (int x = 0; x != N; x++)
            {
                for (int y = 0; y != N; y++)
                {
                    Console.Write(distances[y][x] + "\t");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
            */
            int k = N + (worker2.seqToAlign.count() - N);
            int T = N;  //Offset to detect if the tree node is a subnode or original node (Above N, t is a subnode)
            int NN = N; //Number of original nodes
            while (N > 1)
            {
                double smallest = 1000000;
                int indexSmallestX = -1;
                int indexSmallestY = -1;
                for (int x = 0; x != N; x++)
                {
                    for (int y = 0; y != N; y++)
                    {
                        if (x != y)
                        {
                            double sumx = 0.0;
                            double sumy = 0.0;
                            for (int c = 0; c != N; c++)
                            {
                                sumx += distances[x][c];
                                sumy += distances[y][c];
                            }
                            qmatrix[x][y] = (N - 2) * distances[x][y] - (sumx + sumy);
                            if (qmatrix[x][y] <= smallest)
                            {
                                smallest = qmatrix[x][y];
                                indexSmallestX = x;
                                indexSmallestY = y;
                            }
                        }
                    }
                }

                /*   for (int x = 0; x != N; x++)
                   {
                       for (int y = 0; y != N; y++)
                       {
                           Console.Write(qmatrix[x][y] + "\t");
                       }
                       Console.WriteLine();
                   }
                       */

                Console.WriteLine("merge: " + headers[indexSmallestX] + " and " + headers[indexSmallestY]);
                worker2.gap(gap(0), gap(1), gap(2));
                if (headers[indexSmallestX] >= NN)
                {
                    if (headers[indexSmallestY] >= NN)
                    {
                        aligning[T - NN] = worker2.alignPSP(aligning[headers[indexSmallestX] - NN], aligning[headers[indexSmallestY] - NN], "simple");
                    }
                    else
                    {
                        aligning[T - NN] = worker2.alignPSP(aligning[headers[indexSmallestX] - NN], allDivided[headers[indexSmallestY]], "simple");
                    }
                }
                else
                {
                    if (headers[indexSmallestY] >= NN)
                    {
                        aligning[T - NN] = worker2.alignPSP(allDivided[headers[indexSmallestX]], aligning[headers[indexSmallestY] - NN], "simple");
                    }
                    else
                    {
                        aligning[T - NN] = worker2.alignPSP(allDivided[headers[indexSmallestX]], allDivided[headers[indexSmallestY]], "simple");
                    }
                }
                T++;
                double sumsmx = 0;
                double sumsmy = 0;
                double di = 0.0;
                for (int c = 0; c != N; c++)
                {
                    sumsmx += distances[indexSmallestX][c];
                    sumsmy += distances[indexSmallestY][c];
                }
                di = .5 * distances[indexSmallestX][indexSmallestY] + ((1.0 / (2.0 * (N - 2.0))) * (Double)(sumsmx - sumsmy));
                for (int c = 0; c != N + 1; c++)
                {
                    if (c != N)
                    {
                        if (c == indexSmallestX) { distances[N][c] = di; distances[c][N] = di; }
                        if (c == indexSmallestY) { distances[N][c] = distances[indexSmallestX][indexSmallestY] - di; distances[c][N] = distances[indexSmallestX][indexSmallestY] - di; }
                        if (c != indexSmallestX && c != indexSmallestY) { distances[N][c] = (.5 * distances[indexSmallestX][indexSmallestY]) + (1 / (2 * (N - 1))) * (sumsmx - sumsmy); distances[c][N] = distances[N][c]; }
                    }
                }
                indexSmallestY = (int)headers[indexSmallestY];
                for (int c = 0; c != N + 1; c++)
                {
                    double t = 0.0;
                    t = distances[0][c];
                    distances[0][c] = distances[indexSmallestX][c];
                    distances[indexSmallestX][c] = t;
                    t = distances[c][0];
                    distances[c][0] = distances[c][indexSmallestX];
                    distances[c][indexSmallestX] = t;
                }
                int te = 0;
                te = headers[0];
                headers[0] = headers[indexSmallestX];
                headers[indexSmallestX] = te;
                for (int c = 0; c != N + 1; c++)
                {
                    if (headers[c] == indexSmallestY) { indexSmallestY = c; break; }
                }
                for (int c = 0; c != N + 1; c++)
                {
                    double t = 0.0;
                    t = distances[1][c];
                    distances[1][c] = distances[indexSmallestY][c];
                    distances[indexSmallestY][c] = t;
                    t = distances[c][1];
                    distances[c][1] = distances[c][indexSmallestY];
                    distances[c][indexSmallestY] = t;
                }
                te = headers[1];
                headers[1] = headers[indexSmallestY];
                headers[indexSmallestY] = te;
                for (int x = 2; x != N + 1; x++)
                {
                    headers[x - 2] = headers[x];
                }
                ++k;
                headers[N - 1] = k;
                for (int x = 2; x != N + 1; x++)
                    for (int y = 2; y != N + 1; y++)
                    {
                        distances[x - 2][y - 2] = distances[x][y];
                    }
                N--;
            }
            return aligning[NN - 2];
        }

        /// <summary>
        /// Align All sequences using Center star alignment, testing all sequences as center
        /// This function uses extra files for saving the results of all tests
        /// </summary>
        /// <param name="submatrix"></param>
        public Sequencer centerStarAllvsAll(String submatrix = "blosum62")
        {
            double[] mins = new double[seqToAlign.count()];

            for (int min = 0; min != seqToAlign.count(); min++)
            {
                StreamWriter res = new StreamWriter(".\\Align" + min + ".fasta", false);
                int[][] gaps = new int[seqToAlign.count()][];
                List<int>[] gap_align = new List<int>[seqToAlign.count()];

                for (int c = 0; c != seqToAlign.count(); c++)
                {
                    gap_align[c] = new List<int>();
                    gap_align[c].Clear();
                }

                int count = 0;
                for (int d = 0; d != seqToAlign.count(); d++)
                {
                    removeGaps(d);
                }
                for (int d = 0; d != seqToAlign.count(); d++)
                {
                    if (min != d)
                    {
                        align2("blosum62", min, d);
                        gap_align[count] = removeGaps(min).ToList();
                    }
                    count++;
                }
                Dictionary<string, int> pass = new Dictionary<string, int>();
                pass.Clear();
                int cont = 0, tr = 0; ;
                for (int c = 0; c != gap_align.Count(); c++)
                {
                    for (int d = 0; d != gap_align[c].Count(); d++)
                    {
                        for (int cc = 0; cc != gap_align.Count(); cc++)
                        {
                            if (!pass.ContainsKey(gap_align[c][d] + "_" + cc) && !gap_align[cc].ToList().Contains(gap_align[c][d]))
                            {
                                if (c != cc)
                                {
                                    cont = 0;
                                    tr = 0;
                                    while (cont + d != gap_align[c][d])
                                    {
                                        if (seqToAlign.sequences.ElementAt(min).ElementAt(tr) != '-') cont++;
                                        tr++;
                                    }

                                    seqToAlign.sequences.ElementAt(cc).Insert(cont, '-');
                                    pass.Add(gap_align[c][d] + "_" + cc, 0);
                                }
                                else
                                {
                                    pass.Add(gap_align[c][d] + "_" + cc, 0);
                                }
                            }
                        }
                    }
                }

                res.Write(";---------------------------------------------------------------------------------------------------------\n");
                res.Write(";Center:[" + min + "]\tSum of pairs:[" + sumOfPairs("simple") + "]\tBlosum: [" + sumOfPairs() + "]\tEntrophy:[" + entropy() + "]\n");
                res.Write(";---------------------------------------------------------------------------------------------------------\n");
                mins[min] = sumOfPairs();
                for (int cc = 0; cc != seqToAlign.count(); cc++)
                {
                    res.Write(seqToAlign.getHeader(cc) + "\n" + seqToAlign.getSequence(cc) + "\n");
                    removeGaps(cc);
                }
                res.Close();
            }

            double coolest = mins[0];
            int k = 0;
            for (int cc = 0; cc != seqToAlign.count(); cc++)
            {
                if (coolest < mins[cc])
                {
                    coolest = mins[cc];
                    k = cc;
                }
            }
            return new Sequencer(".\\Align" + k + ".fasta", true);

            /*
               string sep = "";
               int contador = 0;
               sep = "";
              Console.WriteLine("\t1\t2\t3\t4");
               for (int c = 0; c != seqToAlign.count(); c++)
               {
                   Console.Write(c + " " + sep);
                   for (int d = c + 1; d != seqToAlign.count(); d++)
                   {
                       Console.Write("\t" + string.Format("{0:0.00}", distances[contador]));
                       contador++;
                   }
                   Console.WriteLine();
                   sep += "\t";
               }
           */
        }

        /// <summary>
        /// Propossal for distance between 2 sequences
        /// </summary>
        /// <param name="x">Secuence A</param>
        /// <param name="y">Secuence B</param>
        /// <returns></returns>
        public double distancePropossal(int x = 0, int y = 1, int radious = 3)
        {
            double scor = 0;
            Aligner worker = new Aligner(new Sequencer(seqToAlign.getFasta(), false));

            worker.align2("blosum62", x, y);
            //  worker.seqToAlign.print(0, -1, true);
            //Console.WriteLine("(" + x + "," + y + ")");
            worker.seqToAlign.sequences[x] = worker.seqToAlign.sequences[x].Reverse<char>().ToList<char>();
            worker.seqToAlign.sequences[y] = worker.seqToAlign.sequences[y].Reverse<char>().ToList<char>();

            int k = worker.seqToAlign.sequences[x].Count > worker.seqToAlign.sequences[x].Count ? worker.seqToAlign.sequences[x].Count : worker.seqToAlign.sequences[y].Count;

            int cy = worker.seqToAlign.sequences[y].Count;
            int cx = worker.seqToAlign.sequences[x].Count;
            //Console.WriteLine("-------" + cy + "   " + cx + "   " + gap(0) + "   " + gap(1) + "   " + gap(2));
            double g = 0;
            double h = 0;

            int menor = (cy < cx ? cy : cx);
            if (k != -1)
            {
                for (int c = 0; c != cx; c++)
                {
                    for (int d = c - radious; d != c + radious; d++)
                    {
                        if (d >= 0 && d < cy)
                        {
                            int cc = Convert.ToInt32(worker.seqToAlign.sequences.ElementAt(x).ElementAt(c)) - 65;
                            int dd = Convert.ToInt32(worker.seqToAlign.sequences.ElementAt(y).ElementAt(d)) - 65;
                            int guion = Convert.ToInt32('-') - 65;
                            if (d < c)
                            {
                                if (cc != guion && dd != guion)
                                {
                                    g = Blosum62[cc][dd];
                                }
                                else
                                {
                                    g = gap(0);
                                }
                                g += 4 - gap(1);
                                h = radious - (d - (c - radious)) + 1.0;
                                scor += (h / g);
                            }
                            if (d == c)
                            {
                                if (cc != guion && dd != guion)
                                {
                                    scor += Blosum62[cc][dd];
                                }
                                else
                                {
                                    scor += gap(0);
                                }
                            }
                            if (d > c)
                            {
                                if (cc != guion && dd != guion)
                                {
                                    g = Blosum62[cc][dd];
                                }
                                else
                                {
                                    g = gap(0);
                                }
                                g += 4 - gap(1);
                                h = (d - (c - radious)) + 1.0;
                                scor += (h / g);
                            }
                        }
                    }
                }
            }
            else
            {
                Console.Write("Empty sequence. Check your input file.");
            }
            worker.removeGaps(x);
            worker.removeGaps(y);
            //Console.WriteLine("distancia:" + scor);
            return scor;
        }

        /// <summary>
        /// Get the FengDoolittle distance from 2 aligned sequences seqToAlign[x] and seqToAlign[y] uses the sequences stored in the main set
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public double distanceFengDoolittle2(int x = 0, int y = 1)
        {
            double res = 0;
            string original = ">ShuffledA\n" +
                    String.Concat(shuffle(seqToAlign.sequences[x]).ToArray()) +
                    "\n>ShuffledB\n" +
                    String.Concat(shuffle(seqToAlign.sequences[y]).ToArray()) +
                    "\n";

            Double sran = 0;
            for (int c = 0; c != 75; c++)
            {
                Sequencer se = new Sequencer(original, false);
                Aligner shuffled = new Aligner(se);

                sran += shuffled.SumOfPairs2(0, 1);
                original = ">ShuffledA\n" +
                    String.Concat(shuffle(seqToAlign.sequences[x]).ToArray()) +
                    "\n>ShuffledB\n" +
                    String.Concat(shuffle(seqToAlign.sequences[y]).ToArray()) +
                    "\n";
            }

            sran = sran / 75.0;
            res = (Double)(SumOfPairs2(x, y) - sran);
            res /= (Double)(((Double)(SumOfPairs2(y, y) + SumOfPairs2(x, x)) / 2.0) - sran);
            res = -Math.Log(res);
            return res;
        }

        /// <summary>
        /// Scoring the multiple sequences in the main sequences set using minimum entropy
        /// </summary>
        /// <returns></returns>
        public Double entropy()
        {
            Double sum = 0;
            int c = seqToAlign.count();

            makeProfile();

            for (int x = 0; x != profiles.Count; x++)
            {
                c = profiles.ElementAt(x).Count;
                for (int chars = 0; chars != c; chars++)
                {
                    sum -= (profiles.ElementAt(x).ElementAt(chars).Value * c) * Math.Log(profiles.ElementAt(x).ElementAt(chars).Value, 2);
                }
            }

            return sum;
        }

        /// <summary>
        /// Return the gap scoring value, VAL selects "gap-gap", "gap opening" or "gap mismatch"
        /// </summary>
        /// <param name="val"></param>
        /// <returns></returns>
        public int gap(string val)
        {
            if (val == "gap-gap") { return gapgap; }
            if (val == "gap opening") { return gapopening; }
            if (val == "gap mismatch") { return gapmismatch; }
            return 0;
        }

        /// <summary>
        /// Return the gap scoring value, VAL selects gap-gap(0), gap opening (1) or gap mismatch(2)
        /// </summary>
        /// <param name="val"></param>
        /// <returns></returns>
        public int gap(int val)
        {
            if (val == 0) { return gapgap; }
            if (val == 1) { return gapopening; }
            if (val == 2) { return gapmismatch; }
            return 0;
        }

        /// <summary>
        /// Set the default gap penalty for Gap-Gap (GG) Gap opening (GO) and Gap-NotGap (MIS)
        /// </summary>
        /// <param name="gg"></param>
        /// <param name="mis"></param>
        /// <param name="go"></param>
        public void gap(int gg, int go, int mis)
        {
            gapgap = gg;
            gapopening = go;
            gapmismatch = mis;
        }

        /// <summary>
        /// Initialize the Blosum62 matrix with preloaded values
        /// values are coded by english alfabet, with spaces of gap scores
        /// </summary>
        public void initBlosum62()
        {
            Blosum62 = new int[27][];
            Blosum62[0] = new int[27] { 4, -2, 0, -2, -1, -2, 0, -2, -1, -4, -1, -1, -1, -2, -4, -1, -1, -1, 1, 0, -4, 0, -3, -20, -2, -1, -4 };
            Blosum62[1] = new int[27] { -2, 4, -3, 4, 1, -3, -1, 0, -3, -4, 0, -4, -3, 3, -4, -2, 0, -1, 0, -1, -4, -3, -4, -20, -3, 1, -4 };
            Blosum62[2] = new int[27] { 0, -3, 9, -3, -4, -2, -3, -3, -1, -4, -3, -1, -1, -3, -4, -3, -3, -3, -1, -1, -4, -1, -2, -20, -2, -3, -4 };
            Blosum62[3] = new int[27] { -2, 4, -3, 6, 2, -3, -1, -1, -3, -4, -1, -4, -3, 1, -4, -1, 0, -2, 0, -1, -4, -3, -4, -20, -3, 1, -4 };
            Blosum62[4] = new int[27] { -1, 1, -4, 2, 5, -3, -2, 0, -3, -4, 1, -3, -2, 0, -4, -1, 2, 0, 0, -1, -4, -2, -3, -20, -2, 4, -4 };
            Blosum62[5] = new int[27] { -2, -3, -2, -3, -3, 6, -3, -1, 0, -4, -3, 0, 0, -3, -4, -4, -3, -3, -2, -2, -4, -1, 1, -20, 3, -3, -4 };
            Blosum62[6] = new int[27] { 0, -1, -3, -1, -2, -3, 6, -2, -4, -4, -2, -4, -3, 0, -4, -2, -2, -2, 0, -2, -4, -3, -2, -20, -3, -2, -4 };
            Blosum62[7] = new int[27] { -2, 0, -3, -1, 0, -1, -2, 8, -3, -4, -1, -3, -2, 1, -4, -2, 0, 0, -1, -2, -4, -3, -2, -20, 2, 0, -4 };
            Blosum62[8] = new int[27] { -1, -3, -1, -3, -3, 0, -4, -3, 4, -4, -3, 2, 1, -3, -4, -3, -3, -3, -2, -1, -4, 3, -3, -20, -1, -3, -4 };
            Blosum62[9] = new int[27] { -4, -4, -4, -4, -4, -4, -4, -4, -4, 1, -4, -4, -4, -4, 1, -4, -4, -4, -4, -4, 1, -4, -4, -20, -4, -4, 1 };
            Blosum62[10] = new int[27] { -1, 0, -3, -1, 1, -3, -2, -1, -3, -4, 5, -2, -1, 0, -4, -1, 1, 2, 0, -1, -4, -2, -3, -20, -2, 1, -4 };
            Blosum62[11] = new int[27] { -1, -4, -1, -4, -3, 0, -4, -3, 2, -4, -2, 4, 2, -3, -4, -3, -2, -2, -2, -1, -4, 1, -2, -20, -1, -3, -4 };
            Blosum62[12] = new int[27] { -1, -3, -1, -3, -2, 0, -3, -2, 1, -4, -1, 2, 5, -2, -4, -2, 0, -1, -1, -1, -4, 1, -1, -20, -1, -1, -4 };
            Blosum62[13] = new int[27] { -2, 3, -3, 1, 0, -3, 0, 1, -3, -4, 0, -3, -2, 6, -4, -2, 0, 0, 1, 0, -4, -3, -4, -20, -2, 0, -4 };
            Blosum62[14] = new int[27] { -4, -4, -4, -4, -4, -4, -4, -4, -4, 1, -4, -4, -4, -4, 1, -4, -4, -4, -4, -4, 1, -4, -4, -20, -4, -4, 1 };
            Blosum62[15] = new int[27] { -1, -2, -3, -1, -1, -4, -2, -2, -3, -4, -1, -3, -2, -2, -4, 7, -1, -2, -1, -1, -4, -2, -4, -20, -3, -1, -4 };
            Blosum62[16] = new int[27] { -1, 0, -3, 0, 2, -3, -2, 0, -3, -4, 1, -2, 0, 0, -4, -1, 5, 1, 0, -1, -4, -2, -2, -20, -1, 3, -4 };
            Blosum62[17] = new int[27] { -1, -1, -3, -2, 0, -3, -2, 0, -3, -4, 2, -2, -1, 0, -4, -2, 1, 5, -1, -1, -4, -3, -3, -20, -2, 0, -4 };
            Blosum62[18] = new int[27] { 1, 0, -1, 0, 0, -2, 0, -1, -2, -4, 0, -2, -1, 1, -4, -1, 0, -1, 4, 1, -4, -2, -3, -20, -2, 0, -4 };
            Blosum62[19] = new int[27] { 0, -1, -1, -1, -1, -2, -2, -2, -1, -4, -1, -1, -1, 0, -4, -1, -1, -1, 1, 5, -4, 0, -2, -20, -2, -1, -4 };
            Blosum62[20] = new int[27] { -4, -4, -4, -4, -4, -4, -4, -4, -4, 1, -4, -4, -4, -4, 1, -4, -4, -4, -4, -4, 1, -4, -4, -20, -4, -4, 1 };
            Blosum62[21] = new int[27] { 0, -3, -1, -3, -2, -1, -3, -3, 3, -4, -2, 1, 1, -3, -4, -2, -2, -3, -2, 0, -4, 4, -3, -20, -1, -2, -4 };
            Blosum62[22] = new int[27] { -3, -4, -2, -4, -3, 1, -2, -2, -3, -4, -3, -2, -1, -4, -4, -4, -2, -3, -3, -2, -4, -3, 11, -20, 2, -3, -4 };
            Blosum62[23] = new int[27] { -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20 };
            Blosum62[24] = new int[27] { -2, -3, -2, -3, -2, 3, -3, 2, -1, -4, -2, -1, -1, -2, -4, -3, -1, -2, -2, -2, -4, -1, 2, -20, 7, -2, -4 };
            Blosum62[25] = new int[27] { -1, 1, -3, 1, 4, -3, -2, 0, -3, -4, 1, -3, -1, 0, -4, -1, 3, 0, 0, -1, -4, -2, -3, -20, -2, 4, -4 };
            Blosum62[26] = new int[27] { -4, -4, -4, -4, -4, -4, -4, -4, -4, 1, -4, -4, -4, -4, 1, -4, -4, -4, -4, -4, 1, -4, -4, -20, -4, -4, 1 };
            /*
            for (int xx = 0; xx != 27; xx++)
            {
                for (int yy = 0; yy != 27; yy++)
                {
                    scoreMatrix[yy * 27 + xx] = Blosum62[xx][yy];
                }
            }
            */
            for (int xx = 0; xx != 27; xx++)
            {
                for (int yy = 0; yy != 27; yy++)
                {
                    if (xx == yy)
                    {
                        scoreMatrix[yy * 27 + xx] = 2;
                    }
                    else
                    {
                        scoreMatrix[yy * 27 + xx] = -1;
                    }
                }
            }
        }

        /// <summary>
        /// Generate probability profile from sequences in seqToAlign
        /// </summary>
        public void makeProfile()
        {
            profiles = new List<Dictionary<char, double>>();
            int c = seqToAlign.count();
            int shortest = seqToAlign.shortest();
            Dictionary<Char, Double> probs = new Dictionary<Char, Double>();
            for (int i = 0; i != shortest; i++)
            {
                probs.Clear();
                for (int row = 0; row != c; row++)
                {
                    if (probs.ContainsKey(seqToAlign.sequences.ElementAt(row).ElementAt(i)))
                    {
                        probs[seqToAlign.sequences.ElementAt(row).ElementAt(i)] += 1.0 / c;
                    }
                    else
                    {
                        probs.Add(seqToAlign.sequences.ElementAt(row).ElementAt(i), 1.0 / c);
                    }
                }
                profiles.Add(probs);
            }
        }

        /// <summary>
        /// Generate a random sequence of A-Z values of X length
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public String randomSequence(int x)
        {
            String res = "";
            for (int c = 0; c != x; c++)
            {
                res += (char)randomAminoacid();
            }
            return res;
        }

        public List<Char> shuffle(List<Char> original, int switches = 0)
        {
            List<Char> res = new List<Char>();

            int ran1;
            char temp = '\0';
            int switchs = 0;
            if (switches > 0) { switchs = switches; } else { switchs = original.Count; }
            res.Clear();
            res = original.ToList<Char>();

            for (int s = 0; s != switchs; s++)
            {
                ran1 = ran.Next(switchs);
                if (res[s] != '-' && res[ran1] != '-')
                {
                    temp = res[s];
                    res[s] = res.ElementAt(ran1);
                    res[ran1] = temp;
                }
            }

            return res;
        }

        /// <summary>
        /// Calculate the sum of matches of 2 sequences
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public int simpleDistance(int x, int y)
        {
            int sum = 0;

            int k = seqToAlign.sequences[x].Count > seqToAlign.sequences[x].Count ? seqToAlign.sequences[x].Count : seqToAlign.sequences[y].Count;

            if (k != -1)
            {
                for (int c = 0; c != k; c++)
                {
                    if (c < seqToAlign.sequences.ElementAt(x).Count && c < seqToAlign.sequences.ElementAt(y).Count)
                    {
                        if (seqToAlign.sequences.ElementAt(x).ElementAt(c) != seqToAlign.sequences.ElementAt(y).ElementAt(c))
                        {
                            sum += 1;
                        }
                    }
                }
            }
            else
            {
                Console.Write("Empty sequence. Check your input file.");
            }
            return sum;
        }

        /// <summary>
        /// Score function using a substitution matrix [matrixSub] for pairwise comparison or blank matrixSub for
        /// penalty of S(Letter, dash)=-2, S(Different Letters)=-1, S(Same letters)=1 , S(dashes)=0
        /// Optional parameters: Start, End to manage to score from sequence[Start] to sequence[end]
        /// </summary>
        /// <returns>int with sum of all scoring combinations of columns between sequences</returns>
        public int sumOfPairs(String matrixSub = "Blosum62", int start = 0, int end = -1)
        {
            if (matrixSub == "Blosum62")
            {
                int backGM = gap(2);

                //gap(gap(0), gap(1), -10);

                int sum = 0;
                int totsum = 0;
                int k = seqToAlign.longest();
                for (int c = 0; c != k; c++)
                {
                    sum = 0;
                    for (int seq = 0; seq != seqToAlign.count(); seq++)
                    {
                        for (int seqToCompare = seq; seqToCompare != seqToAlign.count(); seqToCompare++)
                        {
                            if (seqToCompare != seq)
                            {
                                if (c < seqToAlign.sequences.ElementAt(seq).Count && c < seqToAlign.sequences.ElementAt(seqToCompare).Count)
                                {
                                    if (seqToAlign.sequences.ElementAt(seq).ElementAt(c) - 65 < 27 && seqToAlign.sequences.ElementAt(seq).ElementAt(c) - 65 >= 0 && seqToAlign.sequences.ElementAt(seqToCompare).ElementAt(c) - 65 < 27 && seqToAlign.sequences.ElementAt(seqToCompare).ElementAt(c) - 65 >= 0)
                                    {
                                        sum += Blosum62[seqToAlign.sequences.ElementAt(seq).ElementAt(c) - 65][seqToAlign.sequences.ElementAt(seqToCompare).ElementAt(c) - 65];
                                    }
                                    else
                                    {
                                        if (seqToAlign.sequences.ElementAt(seq).ElementAt(c) == seqToAlign.sequences.ElementAt(seqToCompare).ElementAt(c))
                                        {
                                            sum += gap("gap-gap");
                                        }
                                        else
                                        {
                                            sum += gap("mismatch");
                                        }
                                    }
                                }
                                else
                                {
                                    sum += gap("mismatch");
                                }
                            }
                        }
                    }
                    totsum += sum;
                    sum = 0;
                }
                //gap(gap(0), gap(1), backGM);
                return totsum;
            }
            else
            {
                int st = 0, en = 0;
                if (start < 0) { st = 0; } else { st = start; }
                if (en < st || end > seqToAlign.count() || end == -1) { en = seqToAlign.count(); } else { en = end + 1; }

                int sum = 0;
                int totsum = 0;
                int k = seqToAlign.longest();
                if (k != -1)
                {
                    for (int c = 0; c != k; c++)
                    {
                        sum = 0;
                        for (int seq = st; seq != en; seq++)
                        {
                            for (int seqToCompare = seq; seqToCompare != en; seqToCompare++)
                            {
                                if (seqToCompare != seq)
                                {
                                    if (c < seqToAlign.sequences.ElementAt(seq).Count && c < seqToAlign.sequences.ElementAt(seqToCompare).Count)
                                    {
                                        if (seqToAlign.sequences.ElementAt(seq).ElementAt(c) - 65 < 27 && seqToAlign.sequences.ElementAt(seq).ElementAt(c) - 65 >= 0 && seqToAlign.sequences.ElementAt(seqToCompare).ElementAt(c) - 65 < 27 && seqToAlign.sequences.ElementAt(seqToCompare).ElementAt(c) - 65 >= 0)
                                        {
                                            if (seqToAlign.sequences.ElementAt(seq).ElementAt(c) == seqToAlign.sequences.ElementAt(seqToCompare).ElementAt(c))
                                            {
                                                sum += 1;
                                            }
                                            else
                                            {
                                                sum += -1;
                                            }
                                        }
                                        else
                                        {
                                            if (seqToAlign.sequences.ElementAt(seq).ElementAt(c) == seqToAlign.sequences.ElementAt(seqToCompare).ElementAt(c))
                                            {
                                                sum += 0;
                                            }
                                            else
                                            {
                                                sum += -2;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        sum += -2;
                                    }
                                }
                            }
                        }
                        totsum += sum;
                        sum = 0;
                    }
                }
                else
                {
                    Console.Write("Empty sequence. Check your input file.");
                }
                return totsum;
            }
        }

        /// <summary>
        /// Return the Sum of pairs for a column of sequences[A][x] and column [B][y], using [submatrix] substitution matrix.
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="submatrix"></param>
        /// <returns></returns>
        public int sumOfPairs(Sequencer A, Sequencer B, int x, int y, String submatrix = "blosum62")
        {
            int[][] scoreMatrix = new int[27][];
            for (int c = 0; c != 27; c++) scoreMatrix[c] = new int[27];

            //Transform 2D Scorematrix to vector
            if (submatrix.ToUpper() == "BLOSUM62")
            {
                for (int xx = 0; xx != 27; xx++)
                {
                    for (int yy = 0; yy != 27; yy++)
                    {
                        scoreMatrix[xx][yy] = Blosum62[xx][yy];
                    }
                }
            }

            if (submatrix.ToUpper() == "SIMPLE")
            {
                for (int xx = 0; xx != 27; xx++)
                {
                    for (int yy = 0; yy != 27; yy++)
                    {
                        if (xx != yy)
                        {
                            scoreMatrix[xx][yy] = -1;
                        }
                        else
                        {
                            scoreMatrix[xx][yy] = 2;
                        }
                    }
                }
            }

            int sum = 0;

            int ka = A.longest();
            int kb = B.longest();
            int na = A.count();
            int nb = B.count();
            if (x > -1 && y > -1)
            {
               
                    for (int i = 0; i != na; i++)
                    {
                        for (int j = 0; j != nb; j++)
                        {
                            if (A.sequences[i][x] != '-' && B.sequences[j][y] != '-')
                            {
                                sum += scoreMatrix[A.sequences[i][x] - 65][B.sequences[j][y] - 65];
                            }
                            else
                            {
                                if (A.sequences[i][x] == B.sequences[j][y])
                                {
                                    sum += gap(0);
                                }
                                else
                                {
                                    sum += gap(2);
                                }
                            }
                        }
                    }
                
            }
            else
            {
                if (x == -1 && y > -1)
                {
                    for (int i = 0; i != na; i++)
                    {
                        for (int j = 0; j != nb; j++)
                        {
                            if ('-' == B.sequences[j][y])
                            {
                                sum += gap(0);
                            }
                            else
                            {
                                sum += gap(2);
                            }
                        }
                    }
                }
                if (y == -1 && x > -1)
                {
                    for (int i = 0; i != na; i++)
                    {
                        for (int j = 0; j != nb; j++)
                        {
                            if ('-' == A.sequences[i][x])
                            {
                                sum += gap(0);
                            }
                            else
                            {
                                sum += gap(2);
                            }
                        }
                    }
                }
            }
            return sum;
        }

        public int sumOfPairs_prop(Sequencer A, Sequencer B, int x, int y, String submatrix = "blosum62", int r = 3)
        {
            int sum = 0;

            int ka = A.longest();
            int kb = B.longest();
            int na = A.count();
            int nb = B.count();
            if (x > -1 && y > -1)
            {
                if (x < ka && y < kb)
                {
                    for (int i = 0; i != na; i++)
                    {
                        for (int j = 0; j != nb; j++)
                        {
                            int g = 1;
                            for (int d = y - r; d != y + r; d++)
                            {
                                if (d > -1 && d < B.sequences[j].Count)
                                {
                                    if (A.sequences[i][x] != '-' && B.sequences[j][d] != '-')
                                    {
                                        sum += g * Blosum62[A.sequences[i][x] - 65][B.sequences[j][d] - 65];
                                    }
                                    else
                                    {
                                        if (d > -1 && d < B.sequences[j].Count)
                                        {
                                            if (x > 1 && y > 1)
                                            {
                                                if (A.sequences[i][x - 2] != '-' && B.sequences[j][y - 2] != '-')
                                                {
                                                    if (d > -1 && d < B.sequences[j].Count)
                                                    {
                                                        sum += gap(1);
                                                    }
                                                }
                                            }

                                            if (A.sequences[i][x] == B.sequences[j][d])
                                            {
                                                if (d > -1 && d < B.sequences[j].Count)
                                                {
                                                    sum += gap(0);
                                                }
                                            }
                                            else
                                            {
                                                if (d > -1 && d < B.sequences[j].Count)
                                                {
                                                    sum += gap(2);
                                                }
                                            }
                                        }
                                    }
                                }
                                g += g < r ? 1 : -1;
                            }
                        }
                    }
                }
            }
            else
            {
                if (x == -1 && y > -1)
                {
                    for (int i = 0; i != na; i++)
                    {
                        for (int j = 0; j != nb; j++)
                        {
                            int g = 1;
                            for (int d = y - r; d != y + r; d++)
                            {
                                if (d > -1 && d < B.sequences[j].Count)
                                    if ('-' == B.sequences[j][d])
                                    {
                                        sum += g * gap(0);
                                    }
                                    else
                                    {
                                        sum += g * gap(2);
                                    }
                                g += g < r ? 1 : -1;
                            }
                        }
                    }
                }
                if (y == -1 && x > -1)
                {
                    for (int i = 0; i != na; i++)
                    {
                        for (int j = 0; j != nb; j++)
                        {
                            int g = 1;
                            for (int d = x - r; d != x + r; d++)
                            {
                                if (d > -1 && d < A.sequences[i].Count)
                                    if ('-' == A.sequences[i][d])
                                    {
                                        sum += g * gap(0);
                                    }
                                    else
                                    {
                                        sum += g * gap(2);
                                    }
                                g += g < r ? 1 : -1;
                            }
                        }
                    }
                }
            }
            return sum;
        }

        /// <summary>
        /// Score function using penalty of S(Aminoacid, dash)=-2, S(Different aminoacids)=-1, S(Same aminoacid)=1 , S(dashes)=0;
        /// </summary>
        /// <returns>int with sum of all combinations of columns between sequences</returns>
        public int SumOfPairs2(int x, int y)
        {
            int sum = 0;
            int totsum = 0;
            int k = seqToAlign.sequences[x].Count > seqToAlign.sequences[x].Count ? seqToAlign.sequences[x].Count : seqToAlign.sequences[y].Count;

            if (k != -1)
            {
                for (int c = 0; c != k; c++)
                {
                    sum = 0;

                    if (c < seqToAlign.sequences.ElementAt(x).Count && c < seqToAlign.sequences.ElementAt(y).Count)
                    {
                        if (seqToAlign.sequences.ElementAt(x).ElementAt(c) - 65 < 27 && seqToAlign.sequences.ElementAt(x).ElementAt(c) - 65 >= 0 && seqToAlign.sequences.ElementAt(y).ElementAt(c) - 65 < 27 && seqToAlign.sequences.ElementAt(y).ElementAt(c) - 65 >= 0)
                        {
                            if (seqToAlign.sequences.ElementAt(x).ElementAt(c) == seqToAlign.sequences.ElementAt(y).ElementAt(c))
                            {
                                sum += 1;
                            }
                            else
                            {
                                sum += -1;
                            }
                        }
                        else
                        {
                            if (seqToAlign.sequences.ElementAt(x).ElementAt(c) == seqToAlign.sequences.ElementAt(y).ElementAt(c))
                            {
                                sum += 0;
                            }
                            else
                            {
                                sum += -2;
                            }
                        }
                    }
                    else
                    {
                        sum += -2;
                    }

                    totsum += sum;
                    sum = 0;
                }
            }
            else
            {
                Console.Write("Empty sequence. Check your input file.");
            }
            return totsum;
        }

        private int[] changeGaps(int x)
        {
            List<int> res = new List<int>();
            for (int c = 0; c != seqToAlign.sequences.ElementAt(x).Count; c++)
            {
                if (seqToAlign.sequences.ElementAt(x).ElementAt(c) == '-')
                {
                    seqToAlign.sequences.ElementAt(x).RemoveAt(c);
                    seqToAlign.sequences.ElementAt(x).Insert(c, 'X');
                    res.Add(c);
                }
            }
            return res.ToArray<int>();
        }

        /// <summary>
        /// Retrieve the assigned name of a CUDA Kernel from a file, only first occurrence.
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        private string getKernelName(string file, string name)
        {
            string kernelName = "";
            string fi = File.ReadAllText(file);
            string[] lines = fi.Split('\n');
            for (int c = 0; c != lines.Length - 1; c++)
            {
                if (lines[c].IndexOf("entry") != -1)
                {
                    string[] b = lines[c].Split(' ');
                    if (b[2].IndexOf(name) != -1)
                    {
                        kernelName = b[2];
                    }
                }
            }

            return kernelName.Substring(0, kernelName.Length - 2);
        }

        /// <summary>
        /// Generate a random ascii value corresponding to an aminoacid
        /// </summary>
        /// <returns></returns>
        private int randomAminoacid()
        {
            Random r = new Random();
            int k = r.Next(27);
            while (k + 65 == 'O' || k + 65 == 'J' || k + 65 == 'R' || k + 65 == 'U' || k + 65 == 'X' || k + 65 == 'Z') { k = r.Next(); }
            return 65 + r.Next(27);
        }

        /// <summary>
        /// Find gaps into the selected sequence of the main set, and remove them, also, return all the positions of the gaps in an int array
        /// </summary>
        /// <param name="x">Selector of sequences main set</param>
        /// <returns></returns>
        private int[] removeGaps(int x)
        {
            int[] res;
            int d = 0;
            int c = 0;
            for (int j = 0; j != seqToAlign.sequences[x].Count; j++)
            {
                if (seqToAlign.sequences.ElementAt(x).ElementAt(j) == '-' || seqToAlign.sequences.ElementAt(x).ElementAt(j) == 'X') { c++; }
            }
            res = new int[c];
            c = 0;
            for (int j = 0; j != seqToAlign.sequences[x].Count; j++)
            {
                if (seqToAlign.sequences.ElementAt(x).ElementAt(j) == '-' || seqToAlign.sequences.ElementAt(x).ElementAt(j) == 'X')
                {
                    seqToAlign.sequences.ElementAt(x).RemoveAt(j);
                    res[c] = d;
                    j--;
                    c++;
                }
                d++;
            }

            return res;
        }
    }

    /// <summary>
    /// Main Data type to manage sequences
    /// </summary>
    internal class Sequencer
    {
        /// <summary>
        /// List with the information of each sequence. List of strings
        /// </summary>
        public List<String> headers = new List<String>();

        /// <summary>
        /// List of sequences to represent. List of lists of chars
        /// </summary>
        public List<List<Char>> sequences = new List<List<Char>>();  //Main list of sequences

        /// <summary>
        /// Create a blank new representation of sequences
        /// </summary>
        public Sequencer()
        {
            sequences = new List<List<char>>();
            headers = new List<string>();
        }

        /// <summary>
        /// Create a new representation of HOWMANY sequences (default=-1 for all) skiping OFFSET sequences (Default=0) from the beggining of the reading, if INPUT -> ISFILE, read a file from the operating system, if not ISFILE, input is a fasta format string.
        /// Actual implementation can read files from format:
        /// FAS, FASTA, TFA as FASTA format
        /// MSF as GCG program Pileup and by clustalw alignment file format
        /// ALN as ClustaW alignment file format
        /// PaMSA as PaMSA alignment file format
        /// </summary>
        /// <param name="input"></param>
        /// <param name="isFile"></param>
        /// <param name="howMany"></param>
        /// <param name="offset"></param>
        public Sequencer(string input, bool isFile, int howMany = -1, int offset = 0)   //File with path, howMany sequence to read, -1 to read all, offset of sequences for ignore
        {
            String incSeq = "";
            bool diagonales = false;
            int contadorOffset = 0, contadorSecuencias = 0;
            if (isFile)
            {
                try
                {
                    using (StreamReader sr = new StreamReader(input))
                    {
                        if (Path.GetExtension(input).ToUpper() == ".MSF")
                        {
                            Dictionary<String, String> seqs = new Dictionary<String, String>();
                            while (sr.Peek() >= 0)
                            {
                                String line = sr.ReadLine().ToUpper();
                                if (line != "")
                                {
                                    if (diagonales)
                                    {
                                        if (contadorOffset >= offset)
                                        {
                                            if (howMany == -1 || howMany > contadorSecuencias)
                                            {
                                                string ke = "", va = "";
                                                String[] b;
                                                if (line.ElementAt(0) != ' ')
                                                {
                                                    b = line.Split(' ');
                                                    if (b.Count() > 1)
                                                    {
                                                        ke = b[0];
                                                        for (int c = 1; c != b.Count(); c++)
                                                            va += b[c];

                                                        va = va.Replace(" ", "");
                                                        va = va.Replace(".", "-");
                                                        if (ke != "")
                                                        {
                                                            if (seqs.ContainsKey(ke))
                                                            {
                                                                seqs[ke] += va;
                                                            }
                                                            else
                                                            {
                                                                seqs.Add(ke, va);
                                                            }
                                                            contadorSecuencias = seqs.Count;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        else
                                        {
                                            contadorOffset++;
                                        }
                                    }
                                    if (line == "//") { diagonales = true; }
                                }
                            }
                            for (int c = 0; c != seqs.Count; c++)
                            {
                                sequences.Add(seqs.ElementAt(c).Value.ToList());
                                headers.Add(seqs.ElementAt(c).Key);
                            }
                        }
                        if (Path.GetExtension(input).ToUpper() == ".ALN")
                        {
                            diagonales = true;
                            sr.ReadLine();
                            Dictionary<String, String> seqs = new Dictionary<String, String>();
                            while (sr.Peek() >= 0)
                            {
                                String line = sr.ReadLine().ToUpper();
                                if (line != "")
                                {
                                    if (diagonales)
                                    {
                                        if (contadorOffset >= offset)
                                        {
                                            if (howMany == -1 || howMany > contadorSecuencias)
                                            {
                                                string ke, va;

                                                String[] b;
                                                if (line.ElementAt(0) != ' ')
                                                {
                                                    //b = line.Split(' ');
                                                    b = Regex.Split(line, @"\s+");
                                                    if (b.Count() > 1)
                                                    {
                                                        ke = b[0];
                                                        va = b[1];

                                                        va = va.Replace(" ", "");
                                                        ke = ke.Replace(" ", "");
                                                        va = va.Replace(".", "-");
                                                        if (ke != "")
                                                        {
                                                            if (seqs.ContainsKey(ke))
                                                            {
                                                                seqs[ke] += va;
                                                            }
                                                            else
                                                            {
                                                                seqs.Add(ke, va);
                                                            }
                                                        }
                                                        contadorSecuencias = seqs.Count;
                                                    }
                                                }
                                            }
                                        }
                                        else
                                        {
                                            contadorOffset++;
                                        }
                                    }
                                }
                            }
                            for (int c = 0; c != seqs.Count; c++)
                            {
                                sequences.Add(seqs.ElementAt(c).Value.ToList());
                                headers.Add(seqs.ElementAt(c).Key);
                            }
                        }

                        if (Path.GetExtension(input).ToUpper() == ".PAMSA")
                        {
                            Dictionary<String, String> seqs = new Dictionary<String, String>();
                            while (sr.Peek() >= 0)
                            {
                                String line = sr.ReadLine().ToUpper();
                                if (line.Length > 0)
                                    if (line.Substring(0, 1) == ">")
                                    {
                                        String[] b = line.Split(' ');

                                        if (contadorOffset >= offset)
                                        {
                                            if (howMany == -1 || howMany > contadorSecuencias)
                                            {
                                                string ke, va;
                                                if (b.Count() > 1)
                                                {
                                                    ke = b[0];
                                                    va = b[1];

                                                    if (seqs.ContainsKey(ke))
                                                    {
                                                        seqs[ke] += va;
                                                    }
                                                    else
                                                    {
                                                        seqs.Add(ke, va);
                                                    }
                                                    contadorSecuencias = seqs.Count;
                                                }
                                            }
                                        }
                                        else
                                        {
                                            contadorOffset++;
                                        }
                                    }
                            }
                            for (int c = 0; c != seqs.Count; c++)
                            {
                                sequences.Add(seqs.ElementAt(c).Value.ToList());
                                headers.Add(seqs.ElementAt(c).Key);
                            }
                        }

                        if (Path.GetExtension(input).ToUpper() == ".FAS" || Path.GetExtension(input).ToUpper() == ".TFA" || Path.GetExtension(input).ToUpper() == ".FASTA")
                        {
                            while (sr.Peek() >= 0)
                            {
                                String line = sr.ReadLine().ToUpper();
                                if (line != "")
                                {
                                    if (line.Length > 0)
                                        if (line.Substring(0, 1) == ">")
                                        {
                                            if (contadorOffset >= offset)
                                            {
                                                if (howMany == -1 || howMany > contadorSecuencias)
                                                {
                                                    if (incSeq != "")
                                                    {
                                                        incSeq = incSeq.Replace(".", "-");
                                                        sequences.Add(incSeq.ToList());
                                                        incSeq = "";
                                                        contadorSecuencias++;
                                                    }

                                                    headers.Add(line);
                                                }
                                            }
                                            else
                                            {
                                                contadorOffset++;
                                            }
                                        }
                                        else
                                        {
                                            if (howMany == -1 || howMany > contadorSecuencias)
                                            {
                                                if (line.Substring(0, 1) != ";" && line != "")
                                                {
                                                    incSeq += line;
                                                }
                                            }
                                        }
                                }
                            }
                            if (incSeq != "")
                            {
                                sequences.Add(incSeq.ToList());
                                incSeq = "";
                            }
                        }
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine("There is a problem opening the file: ");
                    Console.WriteLine(e.Message);
                }
            }
            else
            {
                String[] lines = input.ToUpper().Split('\n');
                for (int c = 0; c != lines.Length - 1; c++)
                {
                    if (lines[c].Length > 0)
                        if (lines[c].Substring(0, 1) == ">")
                        {
                            if (incSeq != "")
                            {
                                sequences.Add(incSeq.ToList());
                                incSeq = "";
                            }
                            if (contadorOffset >= offset)
                            {
                                if (howMany == -1 || howMany > contadorSecuencias)
                                {
                                    headers.Add(lines[c]);
                                    contadorSecuencias++;
                                }
                            }
                            else
                            {
                                contadorOffset++;
                            }
                        }
                        else
                        {
                            if (lines[c].Substring(0, 1) != ";")
                            {
                                incSeq += lines[c];
                            }
                        }
                }
                if (incSeq != "")
                {
                    sequences.Add(incSeq.ToList());
                    incSeq = "";
                }
            }
        }

        /// <summary>
        /// Add new sequence with HEADER information and SEQUENCE list of characters to the class
        /// </summary>
        /// <param name="header"></param>
        /// <param name="sequence"></param>
        /// <returns></returns>
        public void addSequence(string header, string sequence)
        {
            headers.Add(header);
            sequences.Add(sequence.ToList());
        }

        /// <summary>
        /// Return the number of sequences in the actual representation
        /// </summary>
        public int count()
        {
            return sequences.Count;
        }

        /// <summary>
        /// Delete the sequence with index X
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public void delete(int x)
        {
            sequences.RemoveAt(x);
            headers.RemoveAt(x);
        }

        /// <summary>
        /// Divide all the objects represented in the Sequencer into an array of single Sequencer objects
        /// </summary>
        /// <param name="X"></param>
        public void divide(Sequencer[] X)
        {
            for (int c = 0; c != sequences.Count; c++)
            {
                X[c] = new Sequencer(getHeader(c) + "\n" + getSequence(c) + "\n", false);
            }
        }

        public double stdDeviation()
        {
            double mean = meanLength();
            double varianza = 0;
            for (int c = 0; c != sequences.Count; c++)
            {
                varianza += Math.Pow((sequences[c].Count() - mean), 2.0);
            }
            return Math.Sqrt(varianza / sequences.Count);
        }

        /// <summary>
        /// Mean of length of the sequences
        /// </summary>

        public double meanLength()
        {
            double mean = 0;
            for (int c = 0; c != sequences.Count; c++)
            {
                mean += sequences[c].Count();
            }
            return mean / sequences.Count;
        }

        public string getFasta()
        {
            string s = "";
            for (int c = 0; c != count(); c++)
            {
                s += getHeader(c) + "\n" + getSequence(c) + "\n";
            }
            return s;
        }

        /// <summary>
        /// Return a string with the header of the sequence with index X
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public string getHeader(int x)
        {    //Return a string with the header of sequence X
            if (x >= 0 && x < sequences.Count)
            {
                return headers.ElementAt(x);
            }
            else
            {
                return "";
            }
        }

        /// <summary>
        /// Return the string with the secuence of index X.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public string getSequence(int x)
        {
            if (x >= 0 && x < sequences.Count)
            {
                return new string(sequences.ElementAt(x).ToArray());
            }
            else
            {
                return "";
            }
        }

        /// <summary>
        /// Return the count of elements in the longest sequence.  O(n)
        /// </summary>
        /// <returns></returns>
        public int longest()
        {
            int longest = 0;
            if (sequences.Count > 1)
            {
                longest = sequences[0].Count;
                for (int c = 1; c != sequences.Count - 1; c++)
                {
                    if (longest < sequences[c].Count) { longest = sequences[c].Count; }
                }

                return longest;
            } if (sequences.Count == 1) { return sequences[0].Count; }
            else { return -1; }
        }

        /// <summary>
        /// Print into the console the data stored in the Sequencer
        /// </summary>
        /// <param name="offset">Number of sequences to skip printing from index 0</param>
        /// <param name="maximum">Number of sequences to print</param>
        /// <param name="onlySeq">Flag to print only sequence, or print header plus sequence</param>
        public void print(int offset = 0, int maximum = -1, bool onlySeq = false, string pathIfToFile = "")
        {
            if (pathIfToFile != "")
            {
                StreamWriter res = new StreamWriter(pathIfToFile, false);
                Aligner worker = new Aligner(this);
                string toFile = "";
                //toFile = ";entropy:" + worker.entropy() + " Sum Of Pairs: "+worker.sumOfPairs()+"\n";
                for (int c = 0; c != sequences.Count; c++)
                {
                    if (maximum == -1)
                    {
                        if (onlySeq)
                        {
                            toFile += getSequence(c);
                        }
                        else
                        {
                            toFile += getHeader(c) + "\n";
                            toFile += getSequence(c) + "\n";
                        }
                    }
                    else
                    {
                        if (onlySeq)
                        {
                            toFile += getSequence(c);
                        }
                        else
                        {
                            toFile += getHeader(c) + "\n";
                            toFile += getSequence(c);
                        }
                    }
                }

                res.Write(toFile);
                res.Close();
            }
            else
            {
                for (int c = 0; c != sequences.Count; c++)
                {
                    if (maximum == -1)
                    {
                        if (onlySeq)
                        {
                            Console.WriteLine(getSequence(c));
                        }
                        else
                        {
                            Console.WriteLine(getHeader(c) + "\n" + getSequence(c));
                        }
                    }
                    else
                    {
                        if (onlySeq)
                        {
                            Console.WriteLine(getSequence(c).Substring(offset, maximum));
                        }
                        else
                        {
                            Console.WriteLine(getHeader(c) + "\n" + getSequence(c).Substring(offset, maximum));
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Return the count of elements in the shortest sequence. O(n)
        /// </summary>
        /// <returns></returns>
        public int shortest()
        {
            int shortest = 0;
            if (sequences.Count > 0)
            {
                shortest = sequences[0].Count;
                for (int c = 1; c != sequences.Count; c++)
                {
                    if (shortest > sequences[c].Count) { shortest = sequences[c].Count; }
                }
                return shortest;
            }
            else { return -1; }
        }
    }

    /// <summary>
    /// Main program class
    /// </summary>
    internal class ZProgram
    {

        public static Dictionary<string, int> fill()
        {
            Dictionary<string, int> radios = new Dictionary<string, int>();
            radios.Add("BB11001", 3);
            radios.Add("BB11002", 4);
            radios.Add("BB11003", 3);
            radios.Add("BB11004", 5);
            radios.Add("BB11005", 6);
            radios.Add("BB11006", 4);
            radios.Add("BB11007", 4);
            radios.Add("BB11008", 4);
            radios.Add("BB11009", 6);
            radios.Add("BB11010", 8);
            radios.Add("BB11011", 3);
            radios.Add("BB11012", 3);
            radios.Add("BB11013", 3);
            radios.Add("BB11014", 9);
            radios.Add("BB11015", 4);
            radios.Add("BB11016", 3);
            radios.Add("BB11017", 3);
            radios.Add("BB11018", 10);
            radios.Add("BB11019", 3);
            radios.Add("BB11020", 3);
            radios.Add("BB11021", 3);
            radios.Add("BB11022", 3);
            radios.Add("BB11023", 6);
            radios.Add("BB11024", 5);
            radios.Add("BB11025", 3);
            radios.Add("BB11026", 3);
            radios.Add("BB11027", 7);
            radios.Add("BB11028", 3);
            radios.Add("BB11029", 3);
            radios.Add("BB11030", 3);
            radios.Add("BB11031", 3);
            radios.Add("BB11032", 3);
            radios.Add("BB11033", 3);
            radios.Add("BB11034", 3);
            radios.Add("BB11035", 3);
            radios.Add("BB11036", 5);
            radios.Add("BB11037", 6);
            radios.Add("BB11038", 5);
            radios.Add("BB12001", 4);
            radios.Add("BB12002", 3);
            radios.Add("BB12003", 3);
            radios.Add("BB12004", 3);
            radios.Add("BB12005", 3);
            radios.Add("BB12006", 6);
            radios.Add("BB12007", 3);
            radios.Add("BB12008", 3);
            radios.Add("BB12009", 4);
            radios.Add("BB12010", 6);
            radios.Add("BB12011", 8);
            radios.Add("BB12012", 4);
            radios.Add("BB12013", 6);
            radios.Add("BB12014", 3);
            radios.Add("BB12015", 5);
            radios.Add("BB12016", 5);
            radios.Add("BB12017", 6);
            radios.Add("BB12018", 5);
            radios.Add("BB12019", 4);
            radios.Add("BB12020", 4);
            radios.Add("BB12021", 3);
            radios.Add("BB12022", 5);
            radios.Add("BB12023", 8);
            radios.Add("BB12024", 3);
            radios.Add("BB12025", 3);
            radios.Add("BB12026", 4);
            radios.Add("BB12027", 7);
            radios.Add("BB12028", 4);
            radios.Add("BB12029", 8);
            radios.Add("BB12030", 4);
            radios.Add("BB12031", 8);
            radios.Add("BB12032", 3);
            radios.Add("BB12033", 3);
            radios.Add("BB12034", 3);
            radios.Add("BB12035", 5);
            radios.Add("BB12036", 4);
            radios.Add("BB12037", 7);
            radios.Add("BB12038", 3);
            radios.Add("BB12039", 4);
            radios.Add("BB12040", 5);
            radios.Add("BB12041", 10);
            radios.Add("BB12042", 4);
            radios.Add("BB12043", 5);
            radios.Add("BB12044", 6);
            radios.Add("BB20001", 4);
            radios.Add("BB20002", 3);
            radios.Add("BB20003", 3);
            radios.Add("BB20004", 3);
            radios.Add("BB20005", 3);
            radios.Add("BB20006", 8);
            radios.Add("BB20007", 6);
            radios.Add("BB20008", 3);
            radios.Add("BB20009", 3);
            radios.Add("BB20010", 4);
            radios.Add("BB20011", 6);
            radios.Add("BB20012", 8);
            radios.Add("BB20013", 4);
            radios.Add("BB20014", 6);
            radios.Add("BB20015", 3);
            radios.Add("BB20016", 3);
            radios.Add("BB20017", 6);
            radios.Add("BB20018", 4);
            radios.Add("BB20019", 9);
            radios.Add("BB20020", 5);
            radios.Add("BB20021", 8);
            radios.Add("BB20022", 4);
            radios.Add("BB20023", 4);
            radios.Add("BB20024", 3);
            radios.Add("BB20025", 8);
            radios.Add("BB20026", 3);
            radios.Add("BB20027", 3);
            radios.Add("BB20028", 3);
            radios.Add("BB20029", 3);
            radios.Add("BB20030", 4);
            radios.Add("BB20031", 3);
            radios.Add("BB20032", 4);
            radios.Add("BB20033", 10);
            radios.Add("BB20034", 3);
            radios.Add("BB20035", 3);
            radios.Add("BB20036", 3);
            radios.Add("BB20037", 9);
            radios.Add("BB20038", 9);
            radios.Add("BB20039", 8);
            radios.Add("BB20040", 3);
            radios.Add("BB20041", 3);
            radios.Add("BB30001", 10);
            radios.Add("BB30002", 9);
            radios.Add("BB30003", 3);
            radios.Add("BB30004", 3);
            radios.Add("BB30005", 5);
            radios.Add("BB30006", 3);
            radios.Add("BB30007", 3);
            radios.Add("BB30008", 6);
            radios.Add("BB30009", 3);
            radios.Add("BB30010", 8);
            radios.Add("BB30011", 6);
            radios.Add("BB30012", 8);
            radios.Add("BB30013", 3);
            radios.Add("BB30014", 3);
            radios.Add("BB30015", 10);
            radios.Add("BB30016", 3);
            radios.Add("BB30017", 4);
            radios.Add("BB30018", 3);
            radios.Add("BB30019", 6);
            radios.Add("BB30020", 3);
            radios.Add("BB30021", 3);
            radios.Add("BB30022", 3);
            radios.Add("BB30023", 3);
            radios.Add("BB30024", 7);
            radios.Add("BB30025", 3);
            radios.Add("BB30026", 6);
            radios.Add("BB30027", 5);
            radios.Add("BB30028", 10);
            radios.Add("BB30029", 6);
            radios.Add("BB30030", 3);
            radios.Add("BB40001", 3);
            radios.Add("BB40002", 3);
            radios.Add("BB40003", 5);
            radios.Add("BB40004", 7);
            radios.Add("BB40005", 9);
            radios.Add("BB40006", 5);
            radios.Add("BB40007", 6);
            radios.Add("BB40008", 10);
            radios.Add("BB40009", 4);
            radios.Add("BB40010", 4);
            radios.Add("BB40011", 3);
            radios.Add("BB40012", 3);
            radios.Add("BB40013", 3);
            radios.Add("BB40014", 3);
            radios.Add("BB40015", 3);
            radios.Add("BB40016", 3);
            radios.Add("BB40017", 3);
            radios.Add("BB40018", 6);
            radios.Add("BB40019", 5);
            radios.Add("BB40020", 3);
            radios.Add("BB40021", 10);
            radios.Add("BB40022", 7);
            radios.Add("BB40023", 3);
            radios.Add("BB40024", 3);
            radios.Add("BB40025", 4);
            radios.Add("BB40026", 3);
            radios.Add("BB40027", 3);
            radios.Add("BB40028", 5);
            radios.Add("BB40029", 4);
            radios.Add("BB40030", 3);
            radios.Add("BB40031", 5);
            radios.Add("BB40032", 4);
            radios.Add("BB40033", 4);
            radios.Add("BB40034", 6);
            radios.Add("BB40035", 3);
            radios.Add("BB40036", 4);
            radios.Add("BB40037", 3);
            radios.Add("BB40038", 3);
            radios.Add("BB40039", 6);
            radios.Add("BB40040", 3);
            radios.Add("BB40041", 3);
            radios.Add("BB40042", 3);
            radios.Add("BB40043", 3);
            radios.Add("BB40044", 3);
            radios.Add("BB40045", 9);
            radios.Add("BB40046", 3);
            radios.Add("BB40047", 3);
            radios.Add("BB40048", 4);
            radios.Add("BB40049", 3);
            radios.Add("BB50001", 7);
            radios.Add("BB50002", 3);
            radios.Add("BB50003", 3);
            radios.Add("BB50004", 3);
            radios.Add("BB50005", 7);
            radios.Add("BB50006", 3);
            radios.Add("BB50007", 3);
            radios.Add("BB50008", 8);
            radios.Add("BB50009", 6);
            radios.Add("BB50010", 5);
            radios.Add("BB50011", 3);
            radios.Add("BB50012", 3);
            radios.Add("BB50013", 8);
            radios.Add("BB50014", 6);
            radios.Add("BB50015", 3);
            radios.Add("BB50016", 3);
            radios.Add("BBS11001", 3);
            radios.Add("BBS11002", 4);
            radios.Add("BBS11003", 3);
            radios.Add("BBS11004", 3);
            radios.Add("BBS11005", 5);
            radios.Add("BBS11006", 3);
            radios.Add("BBS11007", 9);
            radios.Add("BBS11008", 5);
            radios.Add("BBS11009", 7);
            radios.Add("BBS11010", 4);
            radios.Add("BBS11011", 4);
            radios.Add("BBS11012", 4);
            radios.Add("BBS11013", 3);
            radios.Add("BBS11014", 4);
            radios.Add("BBS11015", 5);
            radios.Add("BBS11016", 9);
            radios.Add("BBS11017", 5);
            radios.Add("BBS11018", 4);
            radios.Add("BBS11019", 7);
            radios.Add("BBS11020", 8);
            radios.Add("BBS11021", 3);
            radios.Add("BBS11022", 6);
            radios.Add("BBS11023", 4);
            radios.Add("BBS11024", 9);
            radios.Add("BBS11025", 6);
            radios.Add("BBS11026", 3);
            radios.Add("BBS11027", 5);
            radios.Add("BBS11028", 3);
            radios.Add("BBS11029", 4);
            radios.Add("BBS11030", 3);
            radios.Add("BBS11031", 8);
            radios.Add("BBS11032", 6);
            radios.Add("BBS11033", 5);
            radios.Add("BBS11034", 7);
            radios.Add("BBS11035", 3);
            radios.Add("BBS11036", 3);
            radios.Add("BBS11037", 4);
            radios.Add("BBS11038", 7);
            radios.Add("BBS12001", 7);
            radios.Add("BBS12002", 4);
            radios.Add("BBS12003", 3);
            radios.Add("BBS12004", 5);
            radios.Add("BBS12005", 7);
            radios.Add("BBS12006", 5);
            radios.Add("BBS12007", 5);
            radios.Add("BBS12008", 3);
            radios.Add("BBS12009", 3);
            radios.Add("BBS12010", 10);
            radios.Add("BBS12011", 7);
            radios.Add("BBS12012", 8);
            radios.Add("BBS12013", 3);
            radios.Add("BBS12014", 3);
            radios.Add("BBS12015", 5);
            radios.Add("BBS12016", 4);
            radios.Add("BBS12017", 6);
            radios.Add("BBS12018", 4);
            radios.Add("BBS12019", 7);
            radios.Add("BBS12020", 4);
            radios.Add("BBS12021", 3);
            radios.Add("BBS12022", 3);
            radios.Add("BBS12023", 3);
            radios.Add("BBS12024", 3);
            radios.Add("BBS12025", 5);
            radios.Add("BBS12026", 3);
            radios.Add("BBS12027", 6);
            radios.Add("BBS12028", 6);
            radios.Add("BBS12029", 5);
            radios.Add("BBS12030", 3);
            radios.Add("BBS12031", 7);
            radios.Add("BBS12032", 5);
            radios.Add("BBS12033", 6);
            radios.Add("BBS12034", 6);
            radios.Add("BBS12035", 4);
            radios.Add("BBS12036", 3);
            radios.Add("BBS12037", 7);
            radios.Add("BBS12038", 4);
            radios.Add("BBS12039", 9);
            radios.Add("BBS12040", 3);
            radios.Add("BBS12041", 3);
            radios.Add("BBS12042", 6);
            radios.Add("BBS12043", 8);
            radios.Add("BBS12044", 3);
            radios.Add("BBS20001", 7);
            radios.Add("BBS20002", 3);
            radios.Add("BBS20003", 10);
            radios.Add("BBS20004", 9);
            radios.Add("BBS20005", 3);
            radios.Add("BBS20006", 7);
            radios.Add("BBS20007", 5);
            radios.Add("BBS20008", 5);
            radios.Add("BBS20009", 3);
            radios.Add("BBS20010", 8);
            radios.Add("BBS20011", 3);
            radios.Add("BBS20012", 10);
            radios.Add("BBS20013", 10);
            radios.Add("BBS20014", 3);
            radios.Add("BBS20015", 3);
            radios.Add("BBS20016", 6);
            radios.Add("BBS20017", 10);
            radios.Add("BBS20018", 6);
            radios.Add("BBS20019", 7);
            radios.Add("BBS20020", 8);
            radios.Add("BBS20021", 4);
            radios.Add("BBS20022", 10);
            radios.Add("BBS20023", 7);
            radios.Add("BBS20024", 3);
            radios.Add("BBS20025", 6);
            radios.Add("BBS20026", 4);
            radios.Add("BBS20027", 7);
            radios.Add("BBS20028", 4);
            radios.Add("BBS20029", 3);
            radios.Add("BBS20030", 10);
            radios.Add("BBS20031", 6);
            radios.Add("BBS20032", 3);
            radios.Add("BBS20033", 9);
            radios.Add("BBS20034", 3);
            radios.Add("BBS20035", 6);
            radios.Add("BBS20036", 3);
            radios.Add("BBS20037", 3);
            radios.Add("BBS20038", 8);
            radios.Add("BBS20039", 5);
            radios.Add("BBS20040", 3);
            radios.Add("BBS20041", 8);
            radios.Add("BBS30001", 7);
            radios.Add("BBS30002", 4);
            radios.Add("BBS30003", 3);
            radios.Add("BBS30004", 6);
            radios.Add("BBS30005", 7);
            radios.Add("BBS30006", 5);
            radios.Add("BBS30007", 5);
            radios.Add("BBS30008", 3);
            radios.Add("BBS30009", 3);
            radios.Add("BBS30010", 9);
            radios.Add("BBS30011", 10);
            radios.Add("BBS30012", 8);
            radios.Add("BBS30013", 4);
            radios.Add("BBS30014", 4);
            radios.Add("BBS30015", 3);
            radios.Add("BBS30016", 3);
            radios.Add("BBS30017", 5);
            radios.Add("BBS30018", 9);
            radios.Add("BBS30019", 4);
            radios.Add("BBS30020", 3);
            radios.Add("BBS30021", 10);
            radios.Add("BBS30022", 3);
            radios.Add("BBS30023", 3);
            radios.Add("BBS30024", 4);
            radios.Add("BBS30025", 3);
            radios.Add("BBS30026", 9);
            radios.Add("BBS30027", 9);
            radios.Add("BBS30028", 6);
            radios.Add("BBS30029", 3);
            radios.Add("BBS30030", 8);
            radios.Add("BBS50001", 4);
            radios.Add("BBS50002", 3);
            radios.Add("BBS50003", 5);
            radios.Add("BBS50005", 4);
            radios.Add("BBS50006", 3);
            radios.Add("BBS50007", 4);
            radios.Add("BBS50008", 5);
            radios.Add("BBS50009", 4);
            radios.Add("BBS50010", 10);
            radios.Add("BBS50011", 4);
            radios.Add("BBS50012", 3);
            radios.Add("BBS50013", 4);
            radios.Add("BBS50014", 7);
            radios.Add("BBS50015", 3);
            radios.Add("BBS50016", 5);
            radios.Add("12asA_1atiA", 3);
            radios.Add("153l_1cnsA", 3);
            radios.Add("16pk_1qpg", 4);
            radios.Add("19hcA_1duwA", 3);
            radios.Add("1a02N_1a3qA", 10);
            radios.Add("1a04A_1dz3A", 9);
            radios.Add("1a04A_1ibjA", 3);
            radios.Add("1a04A_3chy", 7);
            radios.Add("1a0cA_1a0dA", 5);
            radios.Add("1a0cA_1bxbD", 3);
            radios.Add("1a0cA_1ubpC", 10);
            radios.Add("1a0fA_1hqoA", 8);
            radios.Add("1a0fA_1ljrA", 4);
            radios.Add("1a0fA_1pd21", 3);
            radios.Add("1a0hA_5hpgA", 3);
            radios.Add("1a0p_1a36A", 3);
            radios.Add("1a0p_1ae9A", 7);
            radios.Add("1a0tP_2mprA", 7);
            radios.Add("1a12A_1jtdB", 3);
            radios.Add("1a1z_1d2zA", 3);
            radios.Add("1a28A_2prgA", 4);
            radios.Add("1a28A_3erdA", 4);
            radios.Add("1a2vA_1ksiB", 9);
            radios.Add("1a36A_1a41", 3);
            radios.Add("1a3aA_1a6jA", 5);
            radios.Add("1a3c_1tc1A", 9);
            radios.Add("1a3k_1c1lA", 3);
            radios.Add("1a3k_1lcl", 8);
            radios.Add("1a3qA_1gof", 3);
            radios.Add("1a44_1qouB", 7);
            radios.Add("1a49A_1dxeA", 6);
            radios.Add("1a49A_1pkyC", 3);
            radios.Add("1a49A_2tpsA", 3);
            radios.Add("1a4iA_1b0aA", 3);
            radios.Add("1a4iA_1ee9A", 6);
            radios.Add("1a53_1dvjA", 6);
            radios.Add("1a53_1hg3A", 3);
            radios.Add("1a53_1nsj", 3);
            radios.Add("1a53_1qo2A", 3);
            radios.Add("1a62_1mjc", 3);
            radios.Add("1a65A_1aozA", 7);
            radios.Add("1a65A_1nif", 3);
            radios.Add("1a6cA_1bmv1", 3);
            radios.Add("1a6dA_1ass", 3);
            radios.Add("1a6dA_1derA", 7);
            radios.Add("1a6jA_1hynP", 3);
            radios.Add("1a6l_2fdn", 3);
            radios.Add("1a6m_1ash", 3);
            radios.Add("1a6m_1cg5B", 3);
            radios.Add("1a6m_1ewaA", 4);
            radios.Add("1a6m_1flp", 4);
            radios.Add("1a6m_1h97A", 6);
            radios.Add("1a6m_1hlb", 5);
            radios.Add("1a6m_1ithA", 4);
            radios.Add("1a6m_2fal", 7);
            radios.Add("1a6m_2gdm", 6);
            radios.Add("1a6m_2hbg", 6);
            radios.Add("1a6m_2vhbA", 4);
            radios.Add("1a6m_3sdhA", 10);
            radios.Add("1a6o_1buhA", 4);
            radios.Add("1a75A_1b8cB", 4);
            radios.Add("1a7s_1agjA", 3);
            radios.Add("1a7s_2cgaB", 8);
            radios.Add("1a7tA_1e5dA", 3);
            radios.Add("1a7tA_1smlA", 7);
            radios.Add("1a7w_1aoiA", 3);
            radios.Add("1a7w_1aoiB", 3);
            radios.Add("1a7w_1b67B", 3);
            radios.Add("1a7w_1bh9B", 3);
            radios.Add("1a7w_1tafA", 3);
            radios.Add("1a7w_1tafB", 3);
            radios.Add("1a8d_1wba", 3);
            radios.Add("1a8h_1qqtA", 10);
            radios.Add("1a8l_1hyuA", 3);
            radios.Add("1a8o_1qrjB", 3);
            radios.Add("1a8rA_1fb1D", 7);
            radios.Add("1a9nA_1dceA", 4);
            radios.Add("1aba_1h75A", 3);
            radios.Add("1aba_1kte", 3);
            radios.Add("1aba_1qfnA", 8);
            radios.Add("1aba_3grx", 5);
            radios.Add("1abrB_1dqgA", 3);
            radios.Add("1abwA_1cqxA", 4);
            radios.Add("1ac5_1auoA", 3);
            radios.Add("1ac5_1cpy", 5);
            radios.Add("1ad3A_1bpwA", 7);
            radios.Add("1ad3A_1euhA", 6);
            radios.Add("1adeA_1dj3A", 5);
            radios.Add("1adjA_1atiA", 9);
            radios.Add("1adjA_1httD", 7);
            radios.Add("1ae9A_1a36A", 3);
            radios.Add("1ae9A_1aihA", 3);
            radios.Add("1aerA_1aerB", 3);
            radios.Add("1ag4_1amm", 3);
            radios.Add("1agdA_1bqsA", 3);
            radios.Add("1agdA_1iakA", 9);
            radios.Add("1agdB_1iakA", 4);
            radios.Add("1agjA_1cqqA", 8);
            radios.Add("1agqA_1vpfA", 4);
            radios.Add("1ah1_1cid", 3);
            radios.Add("1ah1_1f97A", 3);
            radios.Add("1ah1_1kb5B", 5);
            radios.Add("1ah1_1qfoA", 5);
            radios.Add("1ah1_1tlk", 3);
            radios.Add("1ah1_1vcaA", 3);
            radios.Add("1ah1_2ncm", 3);
            radios.Add("1ah7_1ca1", 3);
            radios.Add("1ah9_1bkb", 3);
            radios.Add("1aihA_1a0p", 5);
            radios.Add("1air_1pcl", 9);
            radios.Add("1aisB_1guxB", 3);
            radios.Add("1aj6_1b63A", 10);
            radios.Add("1aj8A_1csh", 6);
            radios.Add("1ajqA_1cp9A", 3);
            radios.Add("1ajqB_1nedA", 3);
            radios.Add("1ajsA_1aam", 4);
            radios.Add("1ajsA_1ibjA", 3);
            radios.Add("1ak1_1hrkA", 3);
            radios.Add("1ako_1bix", 9);
            radios.Add("1alu_1bgc", 4);
            radios.Add("1alu_1cnt3", 3);
            radios.Add("1alu_1i1rB", 3);
            radios.Add("1aly_1d4vB", 3);
            radios.Add("1aly_1dyoA", 3);
            radios.Add("1aly_2tnfA", 5);
            radios.Add("1am2_1at0", 3);
            radios.Add("1am7A_3lzt", 3);
            radios.Add("1amj_1zymA", 6);
            radios.Add("1amk_1hg3A", 3);
            radios.Add("1amoA_1ddgB", 6);
            radios.Add("1amoA_1ep2B", 5);
            radios.Add("1amoA_1fnc", 4);
            radios.Add("1amoA_1qfjA", 3);
            radios.Add("1amoA_1rcf", 7);
            radios.Add("1amp_1b8oA", 3);
            radios.Add("1amp_1cg2A", 3);
            radios.Add("1amp_1cx8B", 3);
            radios.Add("1amp_1xjo", 7);
            radios.Add("1amuA_1lci", 9);
            radios.Add("1an8_1aw7A", 6);
            radios.Add("1an8_1eu4A", 8);
            radios.Add("1an8_1f77B", 8);
            radios.Add("1an9A_1c0pA", 5);
            radios.Add("1aoa_1bkrA", 3);
            radios.Add("1aocA_1bndA", 4);
            radios.Add("1aoeA_1d1gA", 6);
            radios.Add("1aoeA_1vdrA", 3);
            radios.Add("1aoeA_3dfr", 7);
            radios.Add("1aohA_1anu", 5);
            radios.Add("1aohA_1g1kB", 10);
            radios.Add("1aoiA_1bh9B", 3);
            radios.Add("1aoiA_1tafA", 3);
            radios.Add("1aoiB_1a7w", 3);
            radios.Add("1aoiB_1aoiD", 3);
            radios.Add("1aoiB_1jfiB", 3);
            radios.Add("1aoiB_1tafA", 3);
            radios.Add("1aoiC_1f66G", 7);
            radios.Add("1aoiD_1aoiB", 3);
            radios.Add("1aoiD_1jfiB", 3);
            radios.Add("1aoxA_1atzA", 4);
            radios.Add("1aoxA_1auq", 7);
            radios.Add("1aoxA_1ido", 3);
            radios.Add("1aoy_1b4aA", 3);
            radios.Add("1ap0_1dz1A", 3);
            radios.Add("1ap8_1ejhA", 6);
            radios.Add("1apyA_1ayyC", 3);
            radios.Add("1apyB_2gawD", 5);
            radios.Add("1aq0A_1eokA", 3);
            radios.Add("1aq0A_1ghsB", 7);
            radios.Add("1aqb_1avgI", 3);
            radios.Add("1aqb_1bebA", 9);
            radios.Add("1aqb_1epaA", 4);
            radios.Add("1aqb_1h91A", 6);
            radios.Add("1aqt_1e79H", 3);
            radios.Add("1aquA_1fmlB", 4);
            radios.Add("1aqzA_1de3A", 3);
            radios.Add("1aqzA_9rnt", 3);
            radios.Add("1arb_2hrvA", 3);
            radios.Add("1ash_1a6m", 8);
            radios.Add("1ash_1flp", 7);
            radios.Add("1ash_1ithA", 3);
            radios.Add("1ash_2fal", 4);
            radios.Add("1at3A_1fl1A", 8);
            radios.Add("1at3A_1lay", 7);
            radios.Add("1atg_1ixh", 3);
            radios.Add("1atg_1mrp", 3);
            radios.Add("1atg_1wod", 8);
            radios.Add("1atiA_1g5hA", 7);
            radios.Add("1atiA_1qf6A", 3);
            radios.Add("1atlA_1bkcA", 8);
            radios.Add("1atzA_1auq", 3);
            radios.Add("1atzA_1ido", 3);
            radios.Add("1atzA_2scuB", 3);
            radios.Add("1au7A_2hddA", 10);
            radios.Add("1auiB_1dguA", 5);
            radios.Add("1auk_1fsu", 8);
            radios.Add("1auoA_1brt", 6);
            radios.Add("1auoA_1fj2B", 4);
            radios.Add("1auoA_1i6wA", 3);
            radios.Add("1auoA_1jfrA", 3);
            radios.Add("1auq_1ido", 5);
            radios.Add("1auvA_1iow", 6);
            radios.Add("1auyA_1e57B", 5);
            radios.Add("1auyA_1qqp3", 3);
            radios.Add("1auz_1dciA", 3);
            radios.Add("1avaC_1wba", 7);
            radios.Add("1avgI_1epaA", 3);
            radios.Add("1avmA_3mdsA", 3);
            radios.Add("1avpA_1euvA", 3);
            radios.Add("1aw0_1cc8A", 3);
            radios.Add("1aw8B_1cr5A", 3);
            radios.Add("1awcA_1bc8C", 8);
            radios.Add("1awcA_2hts", 7);
            radios.Add("1awcA_2irfG", 3);
            radios.Add("1awd_1ayfA", 8);
            radios.Add("1awe_1btkA", 3);
            radios.Add("1awe_1dbhA", 3);
            radios.Add("1awe_1faoA", 3);
            radios.Add("1awe_1pls", 3);
            radios.Add("1ax4A_2dkb", 7);
            radios.Add("1ax4A_2tplA", 7);
            radios.Add("1ax8_1f6fA", 3);
            radios.Add("1ax8_1rcb", 3);
            radios.Add("1axiB_1bp3B", 6);
            radios.Add("1axiB_1j7vR", 3);
            radios.Add("1axkA_2nlrA", 3);
            radios.Add("1aye_1qmuA", 3);
            radios.Add("1ayfA_1put", 7);
            radios.Add("1aym1_2hwf1", 8);
            radios.Add("1aym2_1hri2", 7);
            radios.Add("1aym2_1qqp2", 3);
            radios.Add("1aym3_1b35A", 3);
            radios.Add("1aym3_1ihmA", 3);
            radios.Add("1aym3_1rud3", 3);
            radios.Add("1az9_1bn5", 3);
            radios.Add("1az9_1c24A", 8);
            radios.Add("1azpA_1bf4A", 3);
            radios.Add("1azsA_1culB", 9);
            radios.Add("1azsA_1fx2A", 10);
            radios.Add("1b0nA_1r69", 10);
            radios.Add("1b0uA_1jj7A", 10);
            radios.Add("1b0uA_1skyE", 8);
            radios.Add("1b16A_1bdb", 9);
            radios.Add("1b16A_1bsvA", 8);
            radios.Add("1b16A_1e6wA", 8);
            radios.Add("1b16A_1fds", 8);
            radios.Add("1b16A_1he2A", 10);
            radios.Add("1b16A_1xel", 5);
            radios.Add("1b16A_1ybvA", 6);
            radios.Add("1b16A_3chy", 3);
            radios.Add("1b20A_1rgeA", 6);
            radios.Add("1b25A_1aorA", 7);
            radios.Add("1b2pA_1msaA", 4);
            radios.Add("1b34B_1d3bA", 3);
            radios.Add("1b34B_1d3bB", 7);
            radios.Add("1b34B_1i8fA", 3);
            radios.Add("1b35A_1b35B", 3);
            radios.Add("1b35A_1ihmA", 3);
            radios.Add("1b35A_2mev1", 3);
            radios.Add("1b37B_1gpeA", 3);
            radios.Add("1b3aA_1hfgA", 5);
            radios.Add("1b3aA_1tvxA", 10);
            radios.Add("1b3jA_1agdA", 3);
            radios.Add("1b3mA_1b8sA", 3);
            radios.Add("1b4kA_1qmlA", 5);
            radios.Add("1b5fB_1smrA", 3);
            radios.Add("1b5l_1rmi", 8);
            radios.Add("1b64_1fjfJ", 3);
            radios.Add("1b6cB_1ia8A", 4);
            radios.Add("1b6e_1fm5A", 3);
            radios.Add("1b6e_1hyrB", 5);
            radios.Add("1b6g_1bn6A", 9);
            radios.Add("1b6g_1cqwA", 3);
            radios.Add("1b6rA_1gsoA", 6);
            radios.Add("1b6tA_1f9aA", 10);
            radios.Add("1b7eA_1f3iA", 3);
            radios.Add("1b7fA_1b64", 3);
            radios.Add("1b7gO_1drw", 3);
            radios.Add("1b7gO_1gcuA", 3);
            radios.Add("1b7gO_1ofgA", 3);
            radios.Add("1b87A_1i21A", 3);
            radios.Add("1b8aA_1aszA", 7);
            radios.Add("1b8aA_1lylA", 10);
            radios.Add("1b8gA_1bjwA", 10);
            radios.Add("1b8gA_1bw0A", 6);
            radios.Add("1b8gA_1iaxA", 6);
            radios.Add("1b8oA_1cb0A", 7);
            radios.Add("1b8xA_1pd21", 3);
            radios.Add("1b8xA_1pgtA", 3);
            radios.Add("1b93A_1g8mA", 3);
            radios.Add("1b9hA_1ibjA", 9);
            radios.Add("1b9lA_1dhn", 5);
            radios.Add("1b9yC_1a0rP", 4);
            radios.Add("1b9zA_1a49A", 3);
            radios.Add("1ba1_1dkgD", 3);
            radios.Add("1ba1_1glcG", 5);
            radios.Add("1bag_1smd", 3);
            radios.Add("1bak_1btkA", 3);
            radios.Add("1bak_1btn", 3);
            radios.Add("1bak_1dynA", 4);
            radios.Add("1bak_1faoA", 3);
            radios.Add("1bak_1mai", 3);
            radios.Add("1bb9_2semA", 3);
            radios.Add("1bbhA_1jafB", 3);
            radios.Add("1bbhA_2ccyA", 5);
            radios.Add("1bbzA_1ckaA", 3);
            radios.Add("1bbzA_1gcpA", 3);
            radios.Add("1bcfA_1dpsA", 10);
            radios.Add("1bcfA_1eumA", 5);
            radios.Add("1bcfA_1qghA", 3);
            radios.Add("1bcfA_2fha", 7);
            radios.Add("1bcpA_1lt3A", 3);
            radios.Add("1bcpD_1prtF", 3);
            radios.Add("1bcpD_1tiiD", 3);
            radios.Add("1bcpD_3chbD", 3);
            radios.Add("1bd3A_1dkuA", 3);
            radios.Add("1bd3A_1nulA", 3);
            radios.Add("1bdb_1cydA", 5);
            radios.Add("1bdb_1e6wA", 3);
            radios.Add("1bdb_1e7wA", 6);
            radios.Add("1bdb_1hdcA", 10);
            radios.Add("1bdb_1oaa", 6);
            radios.Add("1bdb_1ybvA", 3);
            radios.Add("1bdo_1ghj", 4);
            radios.Add("1bdo_1iyu", 3);
            radios.Add("1bdyA_1rlw", 3);
            radios.Add("1be3A_1be3B", 8);
            radios.Add("1be3A_1ezvA", 8);
            radios.Add("1be3A_1ezvB", 5);
            radios.Add("1be3A_1hr6A", 3);
            radios.Add("1be3B_1bccB", 5);
            radios.Add("1be3B_1ezvB", 5);
            radios.Add("1be3B_1hr6A", 4);
            radios.Add("1be3C_3bccC", 3);
            radios.Add("1bebA_1bj7", 3);
            radios.Add("1bebA_1epaA", 3);
            radios.Add("1bebA_1mup", 3);
            radios.Add("1bec_1iakB", 6);
            radios.Add("1bec_1tvdA", 3);
            radios.Add("1bec_3mcg1", 7);
            radios.Add("1befA_1jxpA", 5);
            radios.Add("1befA_2hrvA", 7);
            radios.Add("1bf2_1ehaA", 3);
            radios.Add("1bfd_1d4oA", 3);
            radios.Add("1bfg_1afcH", 8);
            radios.Add("1bfg_1jlyA", 3);
            radios.Add("1bg2_1f9tA", 8);
            radios.Add("1bglA_1bhgA", 3);
            radios.Add("1bh9B_1jfiB", 3);
            radios.Add("1bh9B_1tafA", 3);
            radios.Add("1bhe_1czfA", 4);
            radios.Add("1bhe_1rmg", 3);
            radios.Add("1bhgA_1qnoA", 6);
            radios.Add("1bhtA_5hpgA", 9);
            radios.Add("1bi0_1fx7B", 7);
            radios.Add("1bi0_1smtA", 3);
            radios.Add("1bi5A_1dd8A", 8);
            radios.Add("1bi5A_1hn9A", 8);
            radios.Add("1bif_1qhfA", 8);
            radios.Add("1bj4A_1bjwA", 3);
            radios.Add("1bj7_1epaA", 5);
            radios.Add("1bj7_1obpA", 8);
            radios.Add("1bjnA_1bt4A", 9);
            radios.Add("1bjwA_1bw0A", 4);
            radios.Add("1bjwA_1cl2A", 3);
            radios.Add("1bjwA_1d2fA", 9);
            radios.Add("1bjx_1qgvA", 3);
            radios.Add("1bk5A_1ialA", 3);
            radios.Add("1bk7A_1bolA", 5);
            radios.Add("1bkjA_1f5vA", 3);
            radios.Add("1bkjA_1vfrA", 3);
            radios.Add("1bkpA_1tlcA", 10);
            radios.Add("1bkrA_1aoa", 3);
            radios.Add("1bkrA_1bhdA", 7);
            radios.Add("1bli_1jdc", 10);
            radios.Add("1bm9A_1cf7B", 4);
            radios.Add("1bmdA_1b8pA", 5);
            radios.Add("1bmdA_1ceqA", 10);
            radios.Add("1bmdA_1hyhA", 7);
            radios.Add("1bmdA_2cmd", 9);
            radios.Add("1bmfG_1mabG", 3);
            radios.Add("1bmlC_1qqrC", 7);
            radios.Add("1bn5_1c24A", 3);
            radios.Add("1bn6A_1cv2A", 6);
            radios.Add("1bndA_1aocA", 3);
            radios.Add("1bndA_1nt3A", 3);
            radios.Add("1bo4A_1cjwA", 3);
            radios.Add("1bo4A_1i21A", 3);
            radios.Add("1bob_1qsmA", 3);
            radios.Add("1bob_1yghA", 3);
            radios.Add("1bolA_1bk7A", 3);
            radios.Add("1booA_1eg2A", 6);
            radios.Add("1boy_1a21A", 9);
            radios.Add("1bp7A_1af5", 4);
            radios.Add("1bpi_1brcI", 3);
            radios.Add("1bpv_1bquA", 5);
            radios.Add("1bpv_2fnbA", 8);
            radios.Add("1bpwA_1euhA", 3);
            radios.Add("1bpwA_1eyyA", 4);
            radios.Add("1bqcA_7a3hA", 4);
            radios.Add("1bqg_1ec7B", 10);
            radios.Add("1bqg_1mucA", 3);
            radios.Add("1bqg_1qumA", 6);
            radios.Add("1bqg_2mnr", 3);
            radios.Add("1bqk_2mtaA", 3);
            radios.Add("1bqsA_1cvsC", 3);
            radios.Add("1bquA_1bj8", 3);
            radios.Add("1br9_1d2bA", 3);
            radios.Add("1br9_1jc7A", 3);
            radios.Add("1brmA_1drw", 3);
            radios.Add("1brmA_1gcuA", 3);
            radios.Add("1brmA_1qrrA", 3);
            radios.Add("1brt_1i6wA", 3);
            radios.Add("1brt_1jfrA", 3);
            radios.Add("1brt_1qlwA", 3);
            radios.Add("1bs0A_1b8gA", 3);
            radios.Add("1bs0A_1elqA", 10);
            radios.Add("1bs9_1cex", 3);
            radios.Add("1bs9_1ei9A", 3);
            radios.Add("1bsvA_1bwsA", 6);
            radios.Add("1bsvA_1bxkA", 4);
            radios.Add("1bsvA_1he2A", 3);
            radios.Add("1bsvA_1xel", 6);
            radios.Add("1bt3A_1lla", 8);
            radios.Add("1btkA_1btn", 3);
            radios.Add("1btkA_1dynA", 3);
            radios.Add("1btkA_1faoA", 3);
            radios.Add("1btkA_1pls", 3);
            radios.Add("1btkA_1rrpB", 3);
            radios.Add("1btl_1e25A", 3);
            radios.Add("1btl_1ei5A", 3);
            radios.Add("1btl_1omeB", 3);
            radios.Add("1btl_1skf", 7);
            radios.Add("1btn_1dynA", 8);
            radios.Add("1btn_1mai", 6);
            radios.Add("1btn_1pls", 3);
            radios.Add("1btn_1qqgA", 3);
            radios.Add("1btn_1rrpB", 3);
            radios.Add("1bu2A_1g3nC", 3);
            radios.Add("1bu2A_1vin", 3);
            radios.Add("1bu7A_1ea1A", 3);
            radios.Add("1burA_1rusB", 5);
            radios.Add("1bv1_1em2A", 3);
            radios.Add("1bv1_3pmgA", 5);
            radios.Add("1bvwA_1dysA", 3);
            radios.Add("1bvwA_1tml", 4);
            radios.Add("1bvzA_1cxlA", 3);
            radios.Add("1bvzA_1ehaA", 3);
            radios.Add("1bvzA_1smaA", 3);
            radios.Add("1bvzA_7taa", 7);
            radios.Add("1bw0A_1d2fA", 9);
            radios.Add("1bw9A_1qorA", 7);
            radios.Add("1bw9A_1qp8A", 3);
            radios.Add("1bx4A_1dgyA", 9);
            radios.Add("1bx4A_1rkd", 5);
            radios.Add("1bxkA_1db3A", 7);
            radios.Add("1bxkA_1eq2A", 6);
            radios.Add("1bxkA_1he2A", 3);
            radios.Add("1bxkA_1xel", 9);
            radios.Add("1byb_1fa2A", 10);
            radios.Add("1byfA_1esl", 3);
            radios.Add("1byfA_1fm5A", 5);
            radios.Add("1byfA_1htn", 3);
            radios.Add("1byfA_1rtm1", 3);
            radios.Add("1byi_1eg7A", 3);
            radios.Add("1byi_1nksA", 3);
            radios.Add("1bykA_1gca", 5);
            radios.Add("1bykA_1qpzA", 7);
            radios.Add("1bykA_1tlfA", 3);
            radios.Add("1bykA_2dri", 3);
            radios.Add("1bylA_1ecsA", 5);
            radios.Add("1bylA_1qipA", 5);
            radios.Add("1bylA_1qtoA", 5);
            radios.Add("1byuA_1ctqA", 5);
            radios.Add("1bywA_1drmA", 3);
            radios.Add("1bywA_1g28D", 3);
            radios.Add("1bywA_3pyp", 3);
            radios.Add("1c05A_1dm9A", 4);
            radios.Add("1c0aA_1adjA", 4);
            radios.Add("1c0aA_1lylA", 5);
            radios.Add("1c0nA_1ecxA", 9);
            radios.Add("1c0nA_1elqA", 10);
            radios.Add("1c20A_1ig6A", 3);
            radios.Add("1c24A_1chmA", 8);
            radios.Add("1c24A_1xgmA", 10);
            radios.Add("1c3cA_1f1oA", 8);
            radios.Add("1c3d_1qsjD", 4);
            radios.Add("1c3oB_1qdlB", 10);
            radios.Add("1c3wA_1brd", 3);
            radios.Add("1c3wA_1jgjA", 4);
            radios.Add("1c4rA_1qu0A", 3);
            radios.Add("1c4rA_1sacA", 3);
            radios.Add("1c4xA_1jfrA", 3);
            radios.Add("1c53_2dvh", 3);
            radios.Add("1c8bA_1cfzA", 5);
            radios.Add("1c8oA_1hleA", 9);
            radios.Add("1c8oA_1ovaA", 5);
            radios.Add("1c8oA_1sek", 8);
            radios.Add("1c8pA_1cto", 3);
            radios.Add("1c9kB_1cbuC", 3);
            radios.Add("1c9kB_1d2nA", 3);
            radios.Add("1c9kB_1esc", 3);
            radios.Add("1c9kB_1g5rA", 3);
            radios.Add("1c9kB_2dhqA", 9);
            radios.Add("1ca1_1ah7", 3);
            radios.Add("1caxB_1dgwA", 3);
            radios.Add("1cb8A_1eguA", 3);
            radios.Add("1cczA_1fltX", 3);
            radios.Add("1cczA_1hngB", 3);
            radios.Add("1cczA_1qfoA", 3);
            radios.Add("1cczA_1tlk", 3);
            radios.Add("1cd31_1cd32", 3);
            radios.Add("1cd8_1hxmD", 3);
            radios.Add("1cd8_1tvdA", 3);
            radios.Add("1cd8_1wit", 3);
            radios.Add("1cdkA_1apmE", 7);
            radios.Add("1cdkA_1csn", 7);
            radios.Add("1cdkA_1hcl", 10);
            radios.Add("1cdkA_1ia8A", 7);
            radios.Add("1ceo_1cz1A", 3);
            radios.Add("1ceo_1edg", 3);
            radios.Add("1ceqA_1hyhA", 4);
            radios.Add("1ceqA_2cmd", 9);
            radios.Add("1ceqA_3ldh", 6);
            radios.Add("1cewI_1eqkA", 3);
            radios.Add("1cf7A_1cf7B", 3);
            radios.Add("1cf9A_1g2iA", 3);
            radios.Add("1cg2A_1cx8B", 3);
            radios.Add("1cg2A_1xjo", 3);
            radios.Add("1cg5B_1flp", 4);
            radios.Add("1cg5B_1gcwC", 10);
            radios.Add("1cg5B_1hlb", 3);
            radios.Add("1cg5B_1ithA", 10);
            radios.Add("1cg5B_2fal", 5);
            radios.Add("1cg5B_2hbg", 4);
            radios.Add("1cg5B_3sdhA", 9);
            radios.Add("1chkA_1l92", 7);
            radios.Add("1chmA_1ihoA", 3);
            radios.Add("1cid_1kb5B", 3);
            radios.Add("1cid_1qa9A", 3);
            radios.Add("1cipA_1ctqA", 3);
            radios.Add("1cipA_1hurA", 7);
            radios.Add("1cjwA_1i21A", 10);
            radios.Add("1cjxA_1qipA", 3);
            radios.Add("1ckeA_1dekA", 3);
            radios.Add("1ckuA_1eytA", 4);
            radios.Add("1ckv_1g10A", 5);
            radios.Add("1cl2A_1qgnH", 6);
            radios.Add("1clc_1tf4A", 3);
            radios.Add("1clqA_1nozB", 3);
            radios.Add("1cmbA_1mjoB", 3);
            radios.Add("1cmoA_1hjbC", 3);
            radios.Add("1cnt3_1bgc", 3);
            radios.Add("1cnt3_1cnt2", 8);
            radios.Add("1cnt3_1lki", 3);
            radios.Add("1cnv_1llo", 5);
            radios.Add("1cnzA_1iso", 6);
            radios.Add("1cof_1cnuA", 10);
            radios.Add("1cozA_1ihoA", 3);
            radios.Add("1cozA_2ts1", 5);
            radios.Add("1cp2A_1eg7A", 4);
            radios.Add("1cp2A_2nipA", 3);
            radios.Add("1cpcA_1allA", 6);
            radios.Add("1cpcA_2hbg", 10);
            radios.Add("1cpq_256bA", 3);
            radios.Add("1cpt_1f4tA", 5);
            radios.Add("1cqkA_1fc1B", 7);
            radios.Add("1cqkA_1hxmD", 7);
            radios.Add("1cqkA_1iakA", 5);
            radios.Add("1cqqA_1havA", 3);
            radios.Add("1cqxA_1dlyA", 3);
            radios.Add("1cr1A_2reb", 5);
            radios.Add("1cr5A_1e32A", 3);
            radios.Add("1cr5A_1qcsA", 3);
            radios.Add("1cr6B_1ehyA", 5);
            radios.Add("1cs6A_1fltX", 3);
            radios.Add("1cs6A_1tit", 3);
            radios.Add("1cs8A_1cv8", 9);
            radios.Add("1cs8A_1qmyA", 10);
            radios.Add("1csh_4ctsA", 9);
            radios.Add("1csn_1ckiA", 7);
            radios.Add("1ct9A_1gdoA", 4);
            radios.Add("1ctj_1e29A", 3);
            radios.Add("1ctj_2mtaC", 8);
            radios.Add("1ctn_1d2kA", 4);
            radios.Add("1cto_1bpv", 4);
            radios.Add("1ctqA_1am4D", 6);
            radios.Add("1ctqA_1d4aA", 3);
            radios.Add("1ctqA_1d5cA", 4);
            radios.Add("1cv8_1dkiA", 4);
            radios.Add("1cviA_1rpt", 4);
            radios.Add("1cvjB_2u1a", 3);
            radios.Add("1cvl_1tca", 3);
            radios.Add("1cvsC_1ev2E", 3);
            radios.Add("1cvsC_1fltX", 3);
            radios.Add("1cvsC_1qfoA", 3);
            radios.Add("1cvsC_1wit", 3);
            radios.Add("1cvsC_2ncm", 3);
            radios.Add("1cwpA_1f15B", 3);
            radios.Add("1cwvA_1qfhA", 3);
            radios.Add("1cwyA_1bag", 3);
            radios.Add("1cx8B_1xjo", 3);
            radios.Add("1cxlA_1cgu", 7);
            radios.Add("1cxlA_7taa", 3);
            radios.Add("1cxqA_1b9dA", 3);
            radios.Add("1cxqA_1c0mC", 3);
            radios.Add("1cy5A_3ygsP", 3);
            radios.Add("1cydA_1e6wA", 3);
            radios.Add("1cydA_1e7wA", 3);
            radios.Add("1cydA_1enp", 8);
            radios.Add("1cydA_1eny", 3);
            radios.Add("1cydA_1fds", 10);
            radios.Add("1cydA_1fjhB", 3);
            radios.Add("1cydA_1gegG", 3);
            radios.Add("1cydA_1h5qL", 4);
            radios.Add("1cydA_1he2A", 7);
            radios.Add("1cydA_1oaa", 4);
            radios.Add("1cydA_1ybvA", 4);
            radios.Add("1cydA_2ae2B", 3);
            radios.Add("1cyo_1cxyA", 4);
            radios.Add("1cyx_1plc", 3);
            radios.Add("1cyx_2occB", 6);
            radios.Add("1czfA_1rmg", 10);
            radios.Add("1cztA_1eut", 3);
            radios.Add("1cztA_1ulo", 3);
            radios.Add("1d0nA_2vil", 8);
            radios.Add("1d1dA_1qrjB", 7);
            radios.Add("1d1gA_1vdrA", 9);
            radios.Add("1d1gA_3dfr", 9);
            radios.Add("1d1rA_2if1", 10);
            radios.Add("1d2fA_1c7nF", 4);
            radios.Add("1d2fA_1c7oH", 5);
            radios.Add("1d2iA_1es8A", 3);
            radios.Add("1d2kA_1ctn", 6);
            radios.Add("1d2kA_1e9lA", 7);
            radios.Add("1d2mA_1pjr", 3);
            radios.Add("1d2nA_1g6oA", 10);
            radios.Add("1d2sA_1dykA", 3);
            radios.Add("1d2sA_1sacA", 3);
            radios.Add("1d2tA_1eoiA", 4);
            radios.Add("1d2zA_1d2zB", 6);
            radios.Add("1d2zA_1fadA", 10);
            radios.Add("1d2zA_1ngr", 7);
            radios.Add("1d2zB_1fadA", 7);
            radios.Add("1d2zB_1ngr", 3);
            radios.Add("1d3bA_1d3bB", 3);
            radios.Add("1d3bA_1i8fA", 3);
            radios.Add("1d3bB_1i8fA", 3);
            radios.Add("1d3gA_2dorA", 8);
            radios.Add("1d4oA_1j8fA", 3);
            radios.Add("1d4tA_1csyA", 8);
            radios.Add("1d4tA_1jwoA", 7);
            radios.Add("1d4vA_1extA", 3);
            radios.Add("1d4vB_2tnfA", 3);
            radios.Add("1d5rA_1fpzA", 3);
            radios.Add("1d5rA_1vhrA", 3);
            radios.Add("1d5yA_1bl0A", 3);
            radios.Add("1d6aB_1dm0A", 8);
            radios.Add("1d6jA_1f48A", 3);
            radios.Add("1d6jA_1nksA", 3);
            radios.Add("1d6jA_1nstA", 3);
            radios.Add("1d6jA_3tmkA", 3);
            radios.Add("1d7yA_1nhp", 3);
            radios.Add("1d8jA_1d8kA", 3);
            radios.Add("1d9cA_1ekuB", 7);
            radios.Add("1d9eA_1dvjA", 3);
            radios.Add("1d9eA_1fx6B", 10);
            radios.Add("1d9eA_1qr7A", 5);
            radios.Add("1daaA_1ekfA", 7);
            radios.Add("1daaA_1et0A", 6);
            radios.Add("1daaA_1i1mC", 7);
            radios.Add("1dapA_1dssG", 8);
            radios.Add("1dapA_1hyhA", 6);
            radios.Add("1dar_1efcA", 5);
            radios.Add("1db1A_2lbd", 4);
            radios.Add("1db1A_3erdA", 3);
            radios.Add("1db3A_1bg6", 3);
            radios.Add("1db3A_1fds", 7);
            radios.Add("1db3A_1he2A", 3);
            radios.Add("1db3A_1qrrA", 7);
            radios.Add("1dbtA_1dvjA", 4);
            radios.Add("1dbwA_1tmy", 3);
            radios.Add("1dbwA_3chy", 3);
            radios.Add("1dceB_1ft1B", 8);
            radios.Add("1dcfA_1dbwA", 4);
            radios.Add("1dcfA_3chy", 10);
            radios.Add("1dciA_1ef8A", 3);
            radios.Add("1dciA_2dubF", 6);
            radios.Add("1dcs_1bk0", 3);
            radios.Add("1ddbA_2bidA", 3);
            radios.Add("1ddzA_1i6pA", 3);
            radios.Add("1debA_1fe6A", 3);
            radios.Add("1dekA_1gky", 3);
            radios.Add("1dekA_3tmkA", 3);
            radios.Add("1deoA_1wab", 3);
            radios.Add("1derA_1a6dA", 9);
            radios.Add("1dfuP_1feuA", 3);
            radios.Add("1dg9A_1d2aB", 10);
            radios.Add("1dg9A_1jfvA", 3);
            radios.Add("1dgnA_3crd", 3);
            radios.Add("1dhn_1b9lA", 8);
            radios.Add("1dhpA_1nal3", 4);
            radios.Add("1dhpA_2tpsA", 3);
            radios.Add("1dhr_1ybvA", 5);
            radios.Add("1dhx_1ruxA", 9);
            radios.Add("1di0A_1rvv1", 6);
            radios.Add("1di1A_1ps1A", 3);
            radios.Add("1di6A_1ihcA", 7);
            radios.Add("1din_1ei9A", 3);
            radios.Add("1djxB_1rlw", 3);
            radios.Add("1dk4A_1imbA", 8);
            radios.Add("1dk4A_1inp", 8);
            radios.Add("1dk5A_1aow", 3);
            radios.Add("1dkiA_1cv8", 8);
            radios.Add("1dklA_1qfxA", 3);
            radios.Add("1dkuA_1qb7A", 6);
            radios.Add("1dl2A_1hcuA", 6);
            radios.Add("1dl5A_1dusA", 3);
            radios.Add("1dl5A_1fbnA", 3);
            radios.Add("1dl5A_1g6q2", 3);
            radios.Add("1dl5A_1vid", 3);
            radios.Add("1dl5A_1xvaA", 3);
            radios.Add("1dleA_1qq4A", 6);
            radios.Add("1dleA_1svpA", 6);
            radios.Add("1dlyA_1dlwA", 3);
            radios.Add("1dlyA_2gdm", 7);
            radios.Add("1dlyA_2vhbA", 3);
            radios.Add("1dmgA_1ffkC", 3);
            radios.Add("1dmhA_3pchA", 4);
            radios.Add("1dmhA_3pchM", 5);
            radios.Add("1dn1A_1dcfA", 3);
            radios.Add("1do0A_1e32A", 5);
            radios.Add("1do0A_1g8pA", 4);
            radios.Add("1dokA_1tvxA", 10);
            radios.Add("1dosA_1zen", 5);
            radios.Add("1dpe_1jevA", 9);
            radios.Add("1dpsA_1qghA", 7);
            radios.Add("1dptA_1ca7A", 3);
            radios.Add("1dptA_1otgA", 3);
            radios.Add("1dpuA_1hstA", 3);
            radios.Add("1dpuA_1qbjA", 3);
            radios.Add("1dpuA_1smtA", 3);
            radios.Add("1dqaA_1qaxA", 8);
            radios.Add("1dqnA_1tc1A", 9);
            radios.Add("1dqrA_1iatA", 3);
            radios.Add("1dqrA_2pgi", 7);
            radios.Add("1dquA_1pymA", 3);
            radios.Add("1dqwA_1dvjA", 3);
            radios.Add("1dqyA_1ei9A", 3);
            radios.Add("1dqyA_1ivyA", 6);
            radios.Add("1dr9A_1i85A", 4);
            radios.Add("1drmA_1ew0A", 3);
            radios.Add("1drmA_3pyp", 3);
            radios.Add("1drw_1gcuA", 3);
            radios.Add("1drw_1id1A", 3);
            radios.Add("1drw_1ofgA", 3);
            radios.Add("1dssG_1ceqA", 10);
            radios.Add("1dssG_1he2A", 3);
            radios.Add("1dt4A_1vih", 3);
            radios.Add("1dt6A_1bu7A", 3);
            radios.Add("1dtyA_2dkb", 7);
            radios.Add("1dtyA_2gsaA", 5);
            radios.Add("1dulA_1hq1A", 4);
            radios.Add("1dun_1dupA", 4);
            radios.Add("1dusA_1i4wA", 3);
            radios.Add("1dusA_2dpmA", 9);
            radios.Add("1dvjA_1fwrA", 3);
            radios.Add("1dvjA_1ho1A", 3);
            radios.Add("1dvjA_2tpsA", 3);
            radios.Add("1dvpA_1elkA", 3);
            radios.Add("1dwnA_1msc", 3);
            radios.Add("1dwnA_1qbeA", 3);
            radios.Add("1dxy_1gdhA", 6);
            radios.Add("1dxy_1psdA", 3);
            radios.Add("1dxy_2dldA", 9);
            radios.Add("1dykA_1a3k", 3);
            radios.Add("1dynA_1mai", 3);
            radios.Add("1dynA_1pls", 3);
            radios.Add("1dyoA_1ulo", 3);
            radios.Add("1dz1A_1ap0", 3);
            radios.Add("1dz3A_1iow", 3);
            radios.Add("1dz3A_3chy", 3);
            radios.Add("1e0cA_1rhs", 3);
            radios.Add("1e15A_1ctn", 8);
            radios.Add("1e15A_1e9lA", 5);
            radios.Add("1e20A_1g5qA", 4);
            radios.Add("1e2tA_1f13A", 3);
            radios.Add("1e2xA_1qbjA", 3);
            radios.Add("1e32A_1e69B", 8);
            radios.Add("1e3jA_1pedA", 5);
            radios.Add("1e3jA_1qr6A", 4);
            radios.Add("1e4fT_1g99A", 10);
            radios.Add("1e54A_2omf", 5);
            radios.Add("1e54A_3prn", 3);
            radios.Add("1e5dA_5nul", 3);
            radios.Add("1e69B_1f2tA", 5);
            radios.Add("1e69B_1f2tB", 4);
            radios.Add("1e69B_1gajA", 7);
            radios.Add("1e69B_1qhlA", 9);
            radios.Add("1e6bA_1aw9", 5);
            radios.Add("1e6bA_1eemA", 7);
            radios.Add("1e6bA_1gnwA", 5);
            radios.Add("1e6wA_1e7wA", 3);
            radios.Add("1e6wA_1enp", 4);
            radios.Add("1e6wA_1fds", 3);
            radios.Add("1e6wA_1fjhB", 3);
            radios.Add("1e6wA_1oaa", 6);
            radios.Add("1e6wA_1ybvA", 5);
            radios.Add("1e70M_1qvbA", 4);
            radios.Add("1e70M_1tr1B", 10);
            radios.Add("1e7wA_1ybvA", 3);
            radios.Add("1e8xA_1e8zA", 4);
            radios.Add("1ea1A_1bu7A", 7);
            radios.Add("1eaf_1c4tA", 4);
            radios.Add("1eaf_3cla", 5);
            radios.Add("1eagA_1smrA", 5);
            radios.Add("1eagA_2er7E", 4);
            radios.Add("1eagA_2rmpA", 6);
            radios.Add("1eagA_3pep", 3);
            radios.Add("1eaiC_1ate", 6);
            radios.Add("1ebmA_1mun", 5);
            radios.Add("1ebuA_1gcuA", 3);
            radios.Add("1ecfB_1gdoA", 9);
            radios.Add("1ecpA_1a2zA", 3);
            radios.Add("1ecpA_1k3fA", 5);
            radios.Add("1ecxA_1elqA", 9);
            radios.Add("1ecxA_2gsaA", 9);
            radios.Add("1eduA_1dvpA", 3);
            radios.Add("1eduA_1hg5A", 3);
            radios.Add("1eemA_1gnwA", 3);
            radios.Add("1eemA_1hqoA", 6);
            radios.Add("1eerA_1f6fA", 3);
            radios.Add("1eerB_1iarB", 3);
            radios.Add("1ef1A_1a5r", 3);
            radios.Add("1ef1A_1gg3C", 6);
            radios.Add("1efdN_1qguA", 3);
            radios.Add("1efuB_1tfe", 6);
            radios.Add("1efvA_1efpA", 9);
            radios.Add("1efvA_1mjhA", 3);
            radios.Add("1eg7A_1hyqA", 3);
            radios.Add("1ehaA_1uok", 3);
            radios.Add("1ehyA_1qo7A", 6);
            radios.Add("1ehyA_2bce", 3);
            radios.Add("1ei5A_3pte", 7);
            radios.Add("1ei9A_1cr6B", 3);
            radios.Add("1ei9A_1cvl", 3);
            radios.Add("1eia_1d1dA", 4);
            radios.Add("1eiwA_1egaA", 3);
            radios.Add("1ej0A_1dhr", 3);
            radios.Add("1ej0A_1fbnA", 5);
            radios.Add("1ejeA_1axj", 3);
            radios.Add("1ejeA_1i0rA", 8);
            radios.Add("1ejhA_1ap8", 7);
            radios.Add("1ekeA_2rn2", 3);
            radios.Add("1ekfA_1et0A", 7);
            radios.Add("1ekrA_1mla", 6);
            radios.Add("1elqA_2dkb", 7);
            radios.Add("1ema_1ggxA", 3);
            radios.Add("1emsA_1erzA", 8);
            radios.Add("1enp_1ej0A", 3);
            radios.Add("1enp_1eny", 9);
            radios.Add("1enp_1qsgA", 5);
            radios.Add("1eny_1ybvA", 9);
            radios.Add("1enp_1ybvA", 4);
            radios.Add("1eokA_1d2kA", 3);
            radios.Add("1ep2B_1fnc", 5);
            radios.Add("1epaA_1bj7", 7);
            radios.Add("1epaA_1mup", 5);
            radios.Add("1epaA_1np1A", 3);
            radios.Add("1eq2A_1xel", 6);
            radios.Add("1eq3A_1pinA", 5);
            radios.Add("1eqkA_1cewI", 3);
            radios.Add("1eqkA_1molA", 5);
            radios.Add("1erv_1thx", 4);
            radios.Add("1erzA_1f89A", 9);
            radios.Add("1esc_1eny", 3);
            radios.Add("1esl_1h8uA", 3);
            radios.Add("1esl_1qddA", 3);
            radios.Add("1esl_1qo3C", 3);
            radios.Add("1esl_1rtm1", 3);
            radios.Add("1etb1_1gkeC", 9);
            radios.Add("1eteA_1hmcB", 4);
            radios.Add("1etpA_1fcdC", 5);
            radios.Add("1eu8A_4mbp", 3);
            radios.Add("1euhA_1eyyA", 6);
            radios.Add("1eumA_1qghA", 3);
            radios.Add("1eumA_2fha", 10);
            radios.Add("1eut_2sli", 3);
            radios.Add("1eut_3sil", 3);
            radios.Add("1evhA_1i7aC", 8);
            radios.Add("1evhA_1qc6A", 3);
            radios.Add("1evsA_1lki", 4);
            radios.Add("1ew4A_1dlxA", 9);
            radios.Add("1ewaA_1flp", 6);
            radios.Add("1ewaA_1ithA", 5);
            radios.Add("1ewaA_2fal", 4);
            radios.Add("1ewxA_1gp1A", 3);
            radios.Add("1ewxA_1jfuA", 5);
            radios.Add("1ex1A_3chy", 5);
            radios.Add("1exg_1xbd", 5);
            radios.Add("1extA_1d4vA", 3);
            radios.Add("1exzA_1hmcB", 7);
            radios.Add("1eyrA_1fwyA", 3);
            radios.Add("1eyrA_1ga8A", 3);
            radios.Add("1eyrA_1h7eA", 7);
            radios.Add("1eyyA_1ad3A", 7);
            radios.Add("1ezvB_1be3B", 4);
            radios.Add("1f13A_1g0dA", 3);
            radios.Add("1f2dA_1oasA", 7);
            radios.Add("1f2nA_1smvA", 6);
            radios.Add("1f2nA_2tbvA", 3);
            radios.Add("1f37A_1b9yC", 3);
            radios.Add("1f3yA_1g0sA", 5);
            radios.Add("1f4tA_1egyA", 3);
            radios.Add("1f4tA_1phd", 7);
            radios.Add("1f4tA_1rom", 4);
            radios.Add("1f53A_1g6eA", 3);
            radios.Add("1f5qB_1guxB", 3);
            radios.Add("1f5wA_1neu", 4);
            radios.Add("1f5xA_1foeA", 3);
            radios.Add("1f7cA_1pbwA", 3);
            radios.Add("1f7cA_1tx4A", 4);
            radios.Add("1f94A_1cds", 6);
            radios.Add("1f94A_3ebx", 3);
            radios.Add("1f97A_1qfoA", 3);
            radios.Add("1f9aA_1hybA", 6);
            radios.Add("1faoA_1dynA", 9);
            radios.Add("1faoA_1ef1A", 3);
            radios.Add("1faoA_1fhoA", 3);
            radios.Add("1faoA_1fhxA", 3);
            radios.Add("1faoA_1mai", 3);
            radios.Add("1faoA_1rrpB", 3);
            radios.Add("1fc6A_1pdr", 4);
            radios.Add("1fcdC_1e29A", 3);
            radios.Add("1fchA_1hxiA", 3);
            radios.Add("1fdo_1dmr", 8);
            radios.Add("1fdr_1a8p", 10);
            radios.Add("1fdr_1fnc", 7);
            radios.Add("1fdr_1qfjA", 7);
            radios.Add("1fds_1he2A", 3);
            radios.Add("1fepA_1by5A", 5);
            radios.Add("1ff9A_1gcuA", 3);
            radios.Add("1ff9A_1id1A", 3);
            radios.Add("1ff9A_2scuA", 7);
            radios.Add("1ffkE_1e7kB", 3);
            radios.Add("1fftC_1ocrC", 3);
            radios.Add("1fggA_1fgxA", 3);
            radios.Add("1fggA_1fr9A", 3);
            radios.Add("1fggA_1ga8A", 3);
            radios.Add("1fgkA_1csn", 7);
            radios.Add("1fgkA_1fgiB", 7);
            radios.Add("1fgkA_1ia8A", 6);
            radios.Add("1fgxA_1foaA", 3);
            radios.Add("1fgxA_1g8oA", 3);
            radios.Add("1fgxA_1ga8A", 3);
            radios.Add("1fh6A_1fh6G", 7);
            radios.Add("1fhoA_1pls", 3);
            radios.Add("1fhoA_1qqgA", 3);
            radios.Add("1fhuA_1mucA", 3);
            radios.Add("1fi2A_1caxB", 3);
            radios.Add("1fi2A_1dzrA", 8);
            radios.Add("1fit_1kpf", 6);
            radios.Add("1fj7A_1cvjB", 3);
            radios.Add("1fjcA_1cvjB", 3);
            radios.Add("1fjfL_1fjfQ", 3);
            radios.Add("1fjhB_1eq2A", 3);
            radios.Add("1fjjA_1qouA", 9);
            radios.Add("1fknA_1hvc", 8);
            radios.Add("1fknA_1lywA", 5);
            radios.Add("1fknA_1smrA", 4);
            radios.Add("1fknA_2rmpA", 5);
            radios.Add("1floA_1ae9A", 4);
            radios.Add("1floA_1aihA", 3);
            radios.Add("1flp_1h97A", 4);
            radios.Add("1flp_1hlb", 3);
            radios.Add("1flp_1ithA", 6);
            radios.Add("1flp_2fal", 3);
            radios.Add("1flp_3sdhA", 10);
            radios.Add("1fltX_1qa9A", 3);
            radios.Add("1fltX_1tit", 3);
            radios.Add("1fltX_2ncm", 3);
            radios.Add("1fm5A_1htn", 6);
            radios.Add("1fm5A_1qddA", 4);
            radios.Add("1fm5A_1qo3C", 3);
            radios.Add("1fmb_1b11A", 5);
            radios.Add("1fmb_2hpeA", 5);
            radios.Add("1fmb_2rmpA", 3);
            radios.Add("1fmk_1hcl", 5);
            radios.Add("1fmtA_2gar", 9);
            radios.Add("1fnc_1fb3B", 9);
            radios.Add("1fnc_1fdr", 8);
            radios.Add("1fnc_1qfjA", 3);
            radios.Add("1fnc_2pia", 3);
            radios.Add("1fo5A_1hyuA", 4);
            radios.Add("1fofA_1e25A", 3);
            radios.Add("1fpqA_1dusA", 5);
            radios.Add("1fpqA_1fp1D", 3);
            radios.Add("1fpzA_1pty", 6);
            radios.Add("1fpzA_1vhrA", 5);
            radios.Add("1fqtA_1rfs", 7);
            radios.Add("1fqtA_1rie", 8);
            radios.Add("1fqvD_1vcbB", 6);
            radios.Add("1fr9A_1fgxA", 3);
            radios.Add("1fr9A_1h7eA", 3);
            radios.Add("1fr9A_1i52A", 3);
            radios.Add("1frb_1qrqA", 8);
            radios.Add("1frpA_1dk4A", 5);
            radios.Add("1fshA_1hstA", 3);
            radios.Add("1ft9A_1qbjA", 3);
            radios.Add("1ft9A_2cgpA", 9);
            radios.Add("1fukA_1qvaA", 7);
            radios.Add("1fvaA_1qd1A", 3);
            radios.Add("1fvkA_1bed", 4);
            radios.Add("1fvzA_1bv1", 8);
            radios.Add("1fwkA_1fi4A", 3);
            radios.Add("1fxd_1vjw", 9);
            radios.Add("1fxkA_1fxkC", 3);
            radios.Add("1fxkC_1fxkA", 3);
            radios.Add("1fzgD_2eboA", 4);
            radios.Add("1g0rA_1qg8A", 3);
            radios.Add("1g10A_1ckv", 6);
            radios.Add("1g1eB_1e91A", 7);
            radios.Add("1g24A_1qs1A", 3);
            radios.Add("1g4uS_1ytw", 3);
            radios.Add("1g4uS_2shpA", 3);
            radios.Add("1g55A_6mhtA", 8);
            radios.Add("1g5rA_1g6oA", 4);
            radios.Add("1g5rA_1g8yA", 3);
            radios.Add("1g5rA_2reb", 10);
            radios.Add("1g61A_1g62A", 4);
            radios.Add("1g6gA_1qu5A", 3);
            radios.Add("1g6q2_1f3lA", 9);
            radios.Add("1g6q2_1vid", 3);
            radios.Add("1g6q2_2admA", 3);
            radios.Add("1g7eA_1prxA", 7);
            radios.Add("1g8fA_1cozA", 3);
            radios.Add("1g8fA_1f9aA", 3);
            radios.Add("1g8jA_2napA", 5);
            radios.Add("1g8jB_1fqtA", 7);
            radios.Add("1g8jB_1rfs", 3);
            radios.Add("1g8jB_1rie", 3);
            radios.Add("1g8oA_1qg8A", 3);
            radios.Add("1g8qA_1g8qB", 6);
            radios.Add("1g8yA_1g5rA", 3);
            radios.Add("1ga3A_1rcb", 3);
            radios.Add("1ga8A_1fgxA", 3);
            radios.Add("1ga8A_1g8oA", 3);
            radios.Add("1ga8A_1qg8A", 3);
            radios.Add("1gakA_1lis", 3);
            radios.Add("1gaxA_1ile", 9);
            radios.Add("1gc1G_1g9nG", 3);
            radios.Add("1gca_1qpzA", 3);
            radios.Add("1gca_1tlfA", 6);
            radios.Add("1gca_2dri", 6);
            radios.Add("1gceA_3pte", 8);
            radios.Add("1gci_1sud", 4);
            radios.Add("1gdhA_1psdA", 3);
            radios.Add("1gefA_1hh1A", 3);
            radios.Add("1gen_1hxn", 6);
            radios.Add("1gen_1pex", 4);
            radios.Add("1ggqA_1f1mC", 6);
            radios.Add("1ggxA_1h4uA", 3);
            radios.Add("1ghj_1iyu", 3);
            radios.Add("1ghqB_1e5gA", 3);
            radios.Add("1gkxA_1id0A", 6);
            radios.Add("1gky_1nksA", 4);
            radios.Add("1gky_1qhsA", 8);
            radios.Add("1gky_3tmkA", 7);
            radios.Add("1gnwA_1ljrA", 5);
            radios.Add("1gnwA_1pgtA", 4);
            radios.Add("1gp1A_1erv", 4);
            radios.Add("1gp1A_1jfuA", 3);
            radios.Add("1gpl_1bu8A", 3);
            radios.Add("1gpmA_1qdlB", 4);
            radios.Add("1gr2A_1ii5A", 3);
            radios.Add("1gsa_2hgsA", 3);
            radios.Add("1gtxA_2dkb", 6);
            radios.Add("1gtxA_2gsaA", 9);
            radios.Add("1h70A_1jdw", 10);
            radios.Add("1h75A_1kte", 3);
            radios.Add("1h7wA_2dorA", 3);
            radios.Add("1h8cA_1c1yB", 3);
            radios.Add("1h8cA_1ubi", 3);
            radios.Add("1h8uA_1htn", 8);
            radios.Add("1h8uA_1qddA", 6);
            radios.Add("1h8uA_1rtm1", 10);
            radios.Add("1h97A_1ithA", 5);
            radios.Add("1h97A_2fal", 9);
            radios.Add("1h97A_2hbg", 4);
            radios.Add("1h97A_3sdhA", 3);
            radios.Add("1h9jA_1g291", 3);
            radios.Add("1ha1_1f9fA", 3);
            radios.Add("1ha1_1hd0A", 3);
            radios.Add("1ha1_1jmtA", 4);
            radios.Add("1ha1_2u2fA", 3);
            radios.Add("1han_1cjxA", 3);
            radios.Add("1han_1eirA", 3);
            radios.Add("1han_1jc4A", 3);
            radios.Add("1havA_1cqqA", 3);
            radios.Add("1hcl_1csn", 3);
            radios.Add("1hd2A_1qq2A", 4);
            radios.Add("1hdmB_1b3jA", 10);
            radios.Add("1he2A_1hyhA", 3);
            radios.Add("1he2A_1qrrA", 3);
            radios.Add("1hg3A_1nsj", 10);
            radios.Add("1hg3A_2tpsA", 3);
            radios.Add("1hgeB_1flcB", 3);
            radios.Add("1hjp_1bvsA", 5);
            radios.Add("1hlb_1dlyA", 3);
            radios.Add("1hlb_2fal", 6);
            radios.Add("1hmcB_1jli", 3);
            radios.Add("1hmt_1lfo", 3);
            radios.Add("1hn9A_1bi5A", 8);
            radios.Add("1hnoA_1nzyA", 3);
            radios.Add("1ho1A_1nsj", 3);
            radios.Add("1hp4A_1qba", 7);
            radios.Add("1hqoA_1g6wA", 6);
            radios.Add("1hqoA_1gnwA", 6);
            radios.Add("1hqoA_1ljrA", 4);
            radios.Add("1hqoA_1pgtA", 7);
            radios.Add("1hs6A_1hyt", 3);
            radios.Add("1hssA_1bea", 9);
            radios.Add("1hssA_1hyp", 3);
            radios.Add("1htn_1ixxB", 3);
            radios.Add("1htn_1rdo1", 5);
            radios.Add("1htn_1rtm1", 3);
            radios.Add("1huuA_1hueA", 3);
            radios.Add("1huw_1a22A", 5);
            radios.Add("1hv8A_1c9kB", 8);
            radios.Add("1hwyA_1b26F", 10);
            radios.Add("1hwyA_1k89", 4);
            radios.Add("1hx1B_2a3dA", 3);
            radios.Add("1hxmA_1tvdA", 3);
            radios.Add("1hxmD_1tvdA", 3);
            radios.Add("1hyhA_1hygB", 4);
            radios.Add("1hyhA_1ldnF", 7);
            radios.Add("1hyhA_2cmd", 9);
            radios.Add("1hyhA_3ldh", 4);
            radios.Add("1hyt_1ezm", 3);
            radios.Add("1hyt_1hs6A", 3);
            radios.Add("1hyuA_1tde", 4);
            radios.Add("1i1jA_1ckaA", 4);
            radios.Add("1i21A_1i12D", 3);
            radios.Add("1i21A_1nmtA", 3);
            radios.Add("1i21A_1yghA", 3);
            radios.Add("1i4wA_1vid", 3);
            radios.Add("1i4wA_2dpmA", 3);
            radios.Add("1i5nA_1qspA", 3);
            radios.Add("1i69A_1pda", 3);
            radios.Add("1i6wA_1jfrA", 3);
            radios.Add("1i6wA_1tca", 3);
            radios.Add("1iae_1kuh", 3);
            radios.Add("1iakA_1hdmA", 9);
            radios.Add("1iakA_1tlk", 4);
            radios.Add("1iarB_1cto", 3);
            radios.Add("1ibjA_1bs0A", 3);
            radios.Add("1ibjA_1cs1A", 5);
            radios.Add("1ibzA_1kcw", 3);
            radios.Add("1ibzA_1plc", 8);
            radios.Add("1ibzA_1rcy", 3);
            radios.Add("1iciA_1nbaA", 3);
            radios.Add("1id1A_1eq2A", 3);
            radios.Add("1id1A_1ofgA", 3);
            radios.Add("1ifa_1huw", 3);
            radios.Add("1ifa_1rmi", 9);
            radios.Add("1ig0A_1ig3A", 4);
            radios.Add("1igtB_1fe8J", 5);
            radios.Add("1ihmA_1a6cA", 3);
            radios.Add("1ihp_1bif", 3);
            radios.Add("1ihp_1qfxA", 3);
            radios.Add("1ii5A_1lst", 3);
            radios.Add("1ijqA_1qlgA", 3);
            radios.Add("1im3D_1tvdA", 3);
            radios.Add("1imbA_1jp4A", 7);
            radios.Add("1imbA_1qgxA", 8);
            radios.Add("1iow_1ehiB", 10);
            radios.Add("1iow_1gpmA", 5);
            radios.Add("1iq3A_2cblA", 3);
            radios.Add("1irp_2irtA", 3);
            radios.Add("1iso_1cnzA", 3);
            radios.Add("1itbB_1kb5B", 3);
            radios.Add("1itbB_1tit", 3);
            radios.Add("1itbB_1tlk", 3);
            radios.Add("1itbB_1wit", 3);
            radios.Add("1itbB_2ncm", 3);
            radios.Add("1ithA_1cg5B", 3);
            radios.Add("1ithA_2fal", 4);
            radios.Add("1ithA_3sdhA", 9);
            radios.Add("1ixh_1pot", 3);
            radios.Add("1ixxB_1eggA", 5);
            radios.Add("1iyu_1dczA", 3);
            radios.Add("1iyu_1ghj", 3);
            radios.Add("1j9yA_1oyc", 3);
            radios.Add("1jdc_1amy", 8);
            radios.Add("1jfiB_1tafA", 3);
            radios.Add("1jfiB_1tafB", 3);
            radios.Add("1jfuA_1prxA", 9);
            radios.Add("1jhjA_1xnaA", 3);
            radios.Add("1jhnA_1led", 3);
            radios.Add("1jk0A_1xikA", 9);
            radios.Add("1jkmA_1evqA", 6);
            radios.Add("1jmtA_2u2fA", 3);
            radios.Add("1jn5A_1ounA", 7);
            radios.Add("1jn5B_1opy", 3);
            radios.Add("1jotA_1c3nA", 4);
            radios.Add("1jotA_1vmoA", 3);
            radios.Add("1jp4A_1qgxA", 10);
            radios.Add("1jxpA_1svpA", 7);
            radios.Add("1k89_1bw9A", 10);
            radios.Add("1kb5B_1fo0B", 4);
            radios.Add("1kb5B_1nkr", 3);
            radios.Add("1kb5B_1qa9A", 3);
            radios.Add("1kb5B_1qfoA", 4);
            radios.Add("1kb5B_1tlk", 3);
            radios.Add("1kb5B_1vcaA", 3);
            radios.Add("1kb5B_2ncm", 3);
            radios.Add("1kit_2sli", 3);
            radios.Add("1kjs_1c5a", 4);
            radios.Add("1krs_1b8aA", 3);
            radios.Add("1krs_1lylA", 9);
            radios.Add("1ksiB_1spuB", 9);
            radios.Add("1kuh_1hfc", 9);
            radios.Add("1kuh_1iae", 6);
            radios.Add("1kum_1cxlA", 3);
            radios.Add("1kwaA_1kwaB", 3);
            radios.Add("1kwaA_1pdr", 3);
            radios.Add("1kwaA_1qauA", 8);
            radios.Add("1lam_1ecpA", 3);
            radios.Add("1larA_1d5rA", 3);
            radios.Add("1larA_2shpA", 10);
            radios.Add("1lay_1at3A", 9);
            radios.Add("1lbd_2lbd", 3);
            radios.Add("1lckA_1griA", 3);
            radios.Add("1lcl_1sacA", 3);
            radios.Add("1lea_1qbjA", 3);
            radios.Add("1lfb_2hddA", 8);
            radios.Add("1lfdA_2rgf", 3);
            radios.Add("1lis_1gakA", 3);
            radios.Add("1ljrA_1pd21", 3);
            radios.Add("1lki_1exzA", 6);
            radios.Add("1lla_1hc2", 10);
            radios.Add("1lpbA_1pcn", 3);
            radios.Add("1lst_1nnt", 3);
            radios.Add("1lt3A_1bcpA", 3);
            radios.Add("1lt3A_1xtcA", 3);
            radios.Add("1lvk_1br2F", 4);
            radios.Add("1lvl_1geuB", 3);
            radios.Add("1lxa_1qreA", 3);
            radios.Add("1lxa_2xat", 6);
            radios.Add("1mai_1pls", 3);
            radios.Add("1maz_1f16A", 3);
            radios.Add("1maz_1g5jA", 4);
            radios.Add("1maz_2bidA", 3);
            radios.Add("1mfa_1neu", 3);
            radios.Add("1mfmA_1xsoB", 3);
            radios.Add("1mfmA_2apsB", 4);
            radios.Add("1mgtA_1sfe", 5);
            radios.Add("1mh1_2cmd", 9);
            radios.Add("1mnmA_1c7uB", 3);
            radios.Add("1moq_1bvyF", 3);
            radios.Add("1mpgA_1mun", 3);
            radios.Add("1mpgA_2abk", 7);
            radios.Add("1mrj_1qcjB", 5);
            radios.Add("1mroA_1mroB", 3);
            radios.Add("1mroB_1e6vE", 5);
            radios.Add("1mrp_1d9yA", 6);
            radios.Add("1msc_1qbeA", 8);
            radios.Add("1mspA_1qpxA", 3);
            radios.Add("1mtyB_1mhyB", 3);
            radios.Add("1mtyB_1qq8A", 3);
            radios.Add("1mucA_2mnr", 3);
            radios.Add("1mugA_1bjt", 3);
            radios.Add("1mugA_3eugA", 5);
            radios.Add("1mun_1ebmA", 6);
            radios.Add("1mup_1bj7", 8);
            radios.Add("1mup_1qftA", 3);
            radios.Add("1nar_1d2kA", 3);
            radios.Add("1nbaA_1yacA", 3);
            radios.Add("1nbcA_1g43A", 7);
            radios.Add("1nbcA_1tf4A", 3);
            radios.Add("1ndoA_1rie", 3);
            radios.Add("1nedA_1g3iJ", 4);
            radios.Add("1nedA_1pmaP", 6);
            radios.Add("1neu_1kacB", 4);
            radios.Add("1neu_1tvdA", 10);
            radios.Add("1nf1A_1wer", 6);
            radios.Add("1ng1_1j8yF", 6);
            radios.Add("1nksA_1nstA", 3);
            radios.Add("1nksA_1qhiA", 3);
            radios.Add("1nksA_1shkA", 6);
            radios.Add("1nmtA_1qsmA", 3);
            radios.Add("1nnt_1dsn", 5);
            radios.Add("1nox_1bkjA", 3);
            radios.Add("1npk_1nueD", 3);
            radios.Add("1nseA_1qomB", 10);
            radios.Add("1nsgB_256bA", 3);
            radios.Add("1nukA_1ulo", 3);
            radios.Add("1nwpA_1qhqA", 3);
            radios.Add("1nzyA_1dubE", 7);
            radios.Add("1oaa_1ybvA", 3);
            radios.Add("1oasA_1tdj", 8);
            radios.Add("1obpA_1mup", 5);
            radios.Add("1ocrC_1qleC", 7);
            radios.Add("1opc_1qqiA", 3);
            radios.Add("1opy_3stdA", 3);
            radios.Add("1orc_4croE", 3);
            radios.Add("1otcB_1quqA", 3);
            radios.Add("1ovaA_1sek", 7);
            radios.Add("1ovaA_2achA", 5);
            radios.Add("1oyc_2tmdA", 4);
            radios.Add("1p35A_1i4eA", 9);
            radios.Add("1pauB_1qduL", 3);
            radios.Add("1pcl_1czfA", 3);
            radios.Add("1pcl_1qcxA", 6);
            radios.Add("1pd21_1pgtA", 3);
            radios.Add("1pdgA_1vpfA", 3);
            radios.Add("1pdr_1kwaA", 3);
            radios.Add("1pdr_2pdzA", 3);
            radios.Add("1pedA_1bxzD", 3);
            radios.Add("1pedA_2ohxA", 6);
            radios.Add("1pgtA_1gsdB", 8);
            radios.Add("1pgtA_1gumH", 3);
            radios.Add("1pii_1a53", 4);
            radios.Add("1pjr_1hv8A", 4);
            radios.Add("1pjr_1qvaA", 6);
            radios.Add("1pjr_1uaaA", 4);
            radios.Add("1plc_1rcy", 3);
            radios.Add("1plc_2occB", 6);
            radios.Add("1plq_1axcE", 3);
            radios.Add("1plq_1ge8A", 4);
            radios.Add("1plq_2polA", 3);
            radios.Add("1pls_1foeA", 3);
            radios.Add("1pls_1rrpB", 3);
            radios.Add("1pmaA_1nedA", 5);
            radios.Add("1pmaA_1pmaP", 4);
            radios.Add("1pne_3nul", 3);
            radios.Add("1pot_1lst", 3);
            radios.Add("1poxA_1bfd", 9);
            radios.Add("1preA_1bcpB", 9);
            radios.Add("1prs_1g6eA", 8);
            radios.Add("1prtF_2bosA", 3);
            radios.Add("1prtF_3chbD", 3);
            radios.Add("1pscA_1b5tA", 3);
            radios.Add("1psrA_1cll", 3);
            radios.Add("1psrA_1qlsA", 3);
            radios.Add("1psrA_4icb", 10);
            radios.Add("1ptf_2hid", 6);
            radios.Add("1pty_1ytw", 10);
            radios.Add("1pty_2shpA", 3);
            radios.Add("1pvc4_1ar94", 3);
            radios.Add("1pvl_7ahlA", 3);
            radios.Add("1qa9A_1qfoA", 3);
            radios.Add("1qa9A_1tcrA", 3);
            radios.Add("1qa9A_1tit", 3);
            radios.Add("1qa9A_1tlk", 3);
            radios.Add("1qa9A_1wit", 3);
            radios.Add("1qa9A_2ncm", 3);
            radios.Add("1qazA_1cem", 7);
            radios.Add("1qbhA_1g73D", 3);
            radios.Add("1qbjA_2cgpA", 3);
            radios.Add("1qbzC_2ezoA", 3);
            radios.Add("1qdlB_1i7qB", 6);
            radios.Add("1qdlB_1tmy", 4);
            radios.Add("1qfjA_1fnc", 7);
            radios.Add("1qfoA_1tit", 3);
            radios.Add("1qfoA_1tvdA", 4);
            radios.Add("1qfoA_1vcaA", 3);
            radios.Add("1qfoA_1wit", 3);
            radios.Add("1qfoA_2fcbA", 5);
            radios.Add("1qfoA_2ncm", 3);
            radios.Add("1qghA_1bcfA", 3);
            radios.Add("1qghA_2fha", 3);
            radios.Add("1qguB_3csuA", 4);
            radios.Add("1qgvA_1b9yC", 3);
            radios.Add("1qgvA_1bjx", 3);
            radios.Add("1qgwA_1qgwB", 7);
            radios.Add("1qgwC_1allA", 9);
            radios.Add("1qgwC_1cpcL", 8);
            radios.Add("1qhaA_1g99A", 5);
            radios.Add("1qhiA_1qhsA", 3);
            radios.Add("1qhqA_1bqk", 3);
            radios.Add("1qhsA_3tmkA", 3);
            radios.Add("1qhuA_1ck7A", 3);
            radios.Add("1qhvA_1nobE", 8);
            radios.Add("1qipA_1cjxA", 3);
            radios.Add("1qipA_1fa5B", 6);
            radios.Add("1qj2A_1ffvA", 7);
            radios.Add("1qj2B_1ffvE", 5);
            radios.Add("1qj2C_1ffuF", 5);
            radios.Add("1qj4A_1ei9A", 6);
            radios.Add("1qjdA_1qo8D", 10);
            radios.Add("1qk3A_1gca", 3);
            radios.Add("1qk3A_1nulA", 4);
            radios.Add("1qksA_1n50A", 10);
            radios.Add("1qlaA_1fumA", 3);
            radios.Add("1qlaB_1c1yB", 3);
            radios.Add("1qndA_1c44A", 6);
            radios.Add("1qniA_1ibzA", 8);
            radios.Add("1qnoA_1eceA", 3);
            radios.Add("1qnxA_1cfe", 4);
            radios.Add("1qo0D_1ybvA", 3);
            radios.Add("1qo2A_1thfD", 7);
            radios.Add("1qo3C_1rtm1", 3);
            radios.Add("1qorA_2ohxA", 6);
            radios.Add("1qoxN_1e70M", 6);
            radios.Add("1qpxA_1bf8", 7);
            radios.Add("1qpzA_1tlfA", 5);
            radios.Add("1qpzA_2dri", 4);
            radios.Add("1qq4A_1arb", 3);
            radios.Add("1qq4A_1svpA", 3);
            radios.Add("1qq4A_2hrvA", 7);
            radios.Add("1qq8A_1j77A", 5);
            radios.Add("1qqp1_2mev1", 10);
            radios.Add("1qqp2_1pov1", 3);
            radios.Add("1qqp3_1mec3", 3);
            radios.Add("1qqp3_1pov1", 3);
            radios.Add("1qqp3_2mev1", 3);
            radios.Add("1qr4B_1f97A", 3);
            radios.Add("1qrjB_1ak4C", 5);
            radios.Add("1qs1A_1g24A", 3);
            radios.Add("1qsmA_1yghA", 3);
            radios.Add("1qtrA_1hlgA", 3);
            radios.Add("1qtrA_1qfmA", 5);
            radios.Add("1qu5A_1ygs", 3);
            radios.Add("1qu9A_1qd9C", 5);
            radios.Add("1qupA_1yaiA", 6);
            radios.Add("1quqA_1quqB", 3);
            radios.Add("1r2fA_1xikA", 4);
            radios.Add("1r2fA_1xsm", 10);
            radios.Add("1r69_1b0nA", 3);
            radios.Add("1rcb_1lki", 9);
            radios.Add("1rcb_3inkC", 3);
            radios.Add("1rcf_1d04A", 5);
            radios.Add("1rcf_5nul", 3);
            radios.Add("1rcy_1qhqA", 3);
            radios.Add("1rcy_2cuaA", 3);
            radios.Add("1rec_1bjfB", 5);
            radios.Add("1rfs_1rie", 5);
            radios.Add("1rhoA_1ft3A", 3);
            radios.Add("1rhs_1e0cA", 8);
            radios.Add("1rie_1ezvE", 9);
            radios.Add("1rom_1jipA", 5);
            radios.Add("1rpxA_2dorA", 5);
            radios.Add("1rsy_1a25B", 4);
            radios.Add("1rthA_2rn2", 3);
            radios.Add("1rtm1_1qo3C", 5);
            radios.Add("1rzl_1fk3A", 6);
            radios.Add("1sacA_1b09C", 5);
            radios.Add("1sat_1hfc", 3);
            radios.Add("1sayA_1hzzB", 5);
            radios.Add("1sbp_1wod", 9);
            radios.Add("1scjB_1b64", 3);
            radios.Add("1seiA_1qd7G", 3);
            radios.Add("1sek_1dvmD", 10);
            radios.Add("1sek_1psi", 6);
            radios.Add("1sfe_1eh7A", 5);
            radios.Add("1shcA_1x11A", 3);
            radios.Add("1shkA_1ng1", 3);
            radios.Add("1shkA_3tmkA", 4);
            radios.Add("1shsA_1ejfA", 3);
            radios.Add("1skyB_1a5t", 3);
            radios.Add("1skyE_1e32A", 10);
            radios.Add("1skyE_1skyB", 3);
            radios.Add("1sluA_1fi8E", 10);
            radios.Add("1smrA_2rmpA", 3);
            radios.Add("1smvA_1c8nC", 4);
            radios.Add("1smvA_2tbvA", 3);
            radios.Add("1smvA_4sbvC", 5);
            radios.Add("1sro_1mjc", 8);
            radios.Add("1stfI_1cewI", 3);
            radios.Add("1sur_2nsyA", 3);
            radios.Add("1svpA_5ptp", 3);
            radios.Add("1svy_1svq", 5);
            radios.Add("1svy_2vil", 3);
            radios.Add("1swuA_2aviA", 4);
            radios.Add("1t1dA_1buoA", 9);
            radios.Add("1taq_5ktqA", 9);
            radios.Add("1taxA_1xas", 8);
            radios.Add("1tbgE_1tbgF", 9);
            radios.Add("1tc1A_1qk3A", 6);
            radios.Add("1tcrA_1ah1", 3);
            radios.Add("1tcrA_1qsfD", 7);
            radios.Add("1tfb_1c9bA", 5);
            radios.Add("1tfe_1efuB", 6);
            radios.Add("1tgxA_1cdtA", 6);
            radios.Add("1theB_1ef7A", 7);
            radios.Add("1theB_1qmyA", 4);
            radios.Add("1tig_2ifeA", 3);
            radios.Add("1tit_1wit", 3);
            radios.Add("1tit_2ncm", 7);
            radios.Add("1tlfA_2dri", 5);
            radios.Add("1tlk_1wit", 3);
            radios.Add("1tlk_1wwcA", 3);
            radios.Add("1trb_1f6mF", 4);
            radios.Add("1tul_1dun", 5);
            radios.Add("1tvdA_1bzqN", 3);
            radios.Add("1tvdA_1ivlB", 6);
            radios.Add("1tvdA_1qfwL", 3);
            radios.Add("1tvxA_1a15A", 4);
            radios.Add("1tx4A_1f7cA", 3);
            radios.Add("1tyfA_1nzyA", 6);
            radios.Add("1u2fA_1fjcA", 7);
            radios.Add("1u9aA_1c4zD", 3);
            radios.Add("1u9aA_1i7kB", 3);
            radios.Add("1ubpC_1a4mA", 8);
            radios.Add("1uch_1cmxA", 3);
            radios.Add("1ulo_1cx1A", 3);
            radios.Add("1uok_1bvzA", 7);
            radios.Add("1urnA_1u2fA", 9);
            radios.Add("1urnA_2u1a", 3);
            radios.Add("1uteA_1qhwA", 5);
            radios.Add("1uteA_4kbpA", 3);
            radios.Add("1vcaA_2fcbA", 3);
            radios.Add("1vdrA_3dfr", 4);
            radios.Add("1vfrA_1icuB", 7);
            radios.Add("1vfrA_1nox", 3);
            radios.Add("1vfyA_1hyiA", 6);
            radios.Add("1vid_2admA", 3);
            radios.Add("1vid_2dpmA", 3);
            radios.Add("1vid_3mag", 3);
            radios.Add("1vpfA_1pdgA", 7);
            radios.Add("1vpsA_1dzlA", 4);
            radios.Add("1vsgA_2vsgA", 3);
            radios.Add("1wab_1deoA", 3);
            radios.Add("1wdcC_1br4B", 3);
            radios.Add("1wer_1nf1A", 6);
            radios.Add("1wgjA_2prd", 3);
            radios.Add("1wit_2ncm", 3);
            radios.Add("1wwcA_1he7A", 3);
            radios.Add("1x11A_2nmbA", 3);
            radios.Add("1xbd_1ayoA", 3);
            radios.Add("1xel_1i3nA", 4);
            radios.Add("1xikA_1qghA", 3);
            radios.Add("1xikA_1xsm", 7);
            radios.Add("1xvaA_2dpmA", 3);
            radios.Add("1xvaA_2ercA", 3);
            radios.Add("1xwl_1t7pA", 7);
            radios.Add("1xwl_2kfzA", 8);
            radios.Add("1xxaA_1b4aA", 3);
            radios.Add("1ybvA_1g6q2", 3);
            radios.Add("1ybvA_2ae2B", 3);
            radios.Add("1yer_1b63A", 3);
            radios.Add("1yge_1byt", 9);
            radios.Add("1yghA_1bob", 3);
            radios.Add("1yghA_1i21A", 3);
            radios.Add("1yghA_1qsrA", 3);
            radios.Add("1ygs_1devA", 3);
            radios.Add("1ytbA_1qnaB", 4);
            radios.Add("1zin_5ukdA", 6);
            radios.Add("1zpdA_1qpbB", 4);
            radios.Add("1zxq_1ic1A", 3);
            radios.Add("2a3dA_1fpoA", 3);
            radios.Add("2abk_1ebmA", 4);
            radios.Add("2admA_3mag", 3);
            radios.Add("2afpA_1ixxB", 7);
            radios.Add("2bbkH_1qfmA", 3);
            radios.Add("2bbvA_1f2nA", 4);
            radios.Add("2bce_1acl", 8);
            radios.Add("2bce_1qonA", 7);
            radios.Add("2bopA_1ris", 3);
            radios.Add("2btvA_2btvB", 4);
            radios.Add("2cba_1koqB", 6);
            radios.Add("2cblA_1psrA", 3);
            radios.Add("2cbp_1f56C", 4);
            radios.Add("2ccyA_256bA", 4);
            radios.Add("2cmd_1mldD", 10);
            radios.Add("2cmd_3ldh", 8);
            radios.Add("2cpl_1c5fC", 7);
            radios.Add("2cpl_2nul", 3);
            radios.Add("2dhqA_1c9kB", 9);
            radios.Add("2dkb_2gsaA", 5);
            radios.Add("2dkb_2oatC", 5);
            radios.Add("2dorA_1ep3A", 8);
            radios.Add("2dri_8abp", 4);
            radios.Add("2eboA_1mof", 5);
            radios.Add("2er7E_2rmpA", 6);
            radios.Add("2ercA_3mag", 3);
            radios.Add("2fal_1ebt", 5);
            radios.Add("2fal_3sdhA", 4);
            radios.Add("2fcbA_1iisC", 3);
            radios.Add("2fcbA_2ncm", 3);
            radios.Add("2fnbA_1bpv", 9);
            radios.Add("2gar_1bxkA", 6);
            radios.Add("2gdm_1a6m", 4);
            radios.Add("2gdm_2vhbA", 4);
            radios.Add("2gmfA_1f6fA", 5);
            radios.Add("2hbg_2vhbA", 10);
            radios.Add("2hbg_3sdhA", 9);
            radios.Add("2hddA_1b8iA", 4);
            radios.Add("2hdhA_3hdhB", 8);
            radios.Add("2hrvA_1agjA", 3);
            radios.Add("2hrvA_5ptp", 7);
            radios.Add("2hts_2irfG", 3);
            radios.Add("2i1b_1iltA", 7);
            radios.Add("2i1b_1iraX", 5);
            radios.Add("2i1b_2ila", 10);
            radios.Add("2if1_1d1rA", 8);
            radios.Add("2ila_1abrB", 3);
            radios.Add("2ila_1hce", 8);
            radios.Add("2ilk_1vlk", 3);
            radios.Add("2lbd_2prgA", 10);
            radios.Add("2masA_1ezrA", 8);
            radios.Add("2mbr_1qltA", 3);
            radios.Add("2mev1_1tmf1", 3);
            radios.Add("2nlrA_1h8vF", 4);
            radios.Add("2nmbA_1shcA", 3);
            radios.Add("2nsyA_1sur", 3);
            radios.Add("2occB_1qleB", 3);
            radios.Add("2ohxA_3hudA", 8);
            radios.Add("2omf_1osmC", 9);
            radios.Add("2omf_2por", 9);
            radios.Add("2pgd_1pgjA", 5);
            radios.Add("2pii_1cc8A", 3);
            radios.Add("2pkaA_1a7s", 7);
            radios.Add("2pkaA_1aksA", 3);
            radios.Add("2por_3prn", 5);
            radios.Add("2prgA_3erdA", 3);
            radios.Add("2pth_1c8bA", 10);
            radios.Add("2qwc_1b9vA", 3);
            radios.Add("2qwc_1nsdB", 5);
            radios.Add("2reb_1cr1A", 4);
            radios.Add("2reb_1g19A", 3);
            radios.Add("2rmpA_1avfA", 3);
            radios.Add("2rn2_1ekeA", 3);
            radios.Add("2scuA_1drw", 5);
            radios.Add("2scuA_1eudA", 3);
            radios.Add("2scuB_1eucB", 10);
            radios.Add("2shpA_1i9sA", 3);
            radios.Add("2tbvA_1pov1", 4);
            radios.Add("2tct_1a6i", 3);
            radios.Add("2tgi_1es7C", 3);
            radios.Add("2thiA_1eu8A", 6);
            radios.Add("2tnfA_1d4vB", 7);
            radios.Add("2tpsA_1qpoA", 5);
            radios.Add("2tysA_1pii", 3);
            radios.Add("2tysB_1tdj", 5);
            radios.Add("2u2fA_1u2fA", 4);
            radios.Add("2xat_3tdt", 5);
            radios.Add("3bct_1g3jC", 3);
            radios.Add("3chbD_1prtF", 3);
            radios.Add("3chy_1b00B", 3);
            radios.Add("3chy_1tmy", 3);
            radios.Add("3chy_4tmyB", 3);
            radios.Add("3cla_1nocB", 4);
            radios.Add("3crd_3ygsP", 3);
            radios.Add("3erdA_1dkfA", 5);
            radios.Add("3erdA_1ereB", 6);
            radios.Add("3eugA_1lauE", 3);
            radios.Add("3inkC_1irl", 3);
            radios.Add("3ldh_1hlpA", 7);
            radios.Add("3lzt_1gbzA", 4);
            radios.Add("3nul_1acf", 6);
            radios.Add("3nul_1fil", 3);
            radios.Add("3nul_1ifqA", 3);
            radios.Add("3pchM_3pchA", 3);
            radios.Add("3pte_1pmd", 3);
            radios.Add("3pte_1skf", 5);
            radios.Add("3pyp_1drmA", 3);
            radios.Add("3sdhA_1sctF", 3);
            radios.Add("3seb_1enfA", 9);
            radios.Add("3stdA_2std", 3);
            radios.Add("3tgl_1dt5A", 4);
            radios.Add("3ullA_1kawB", 6);
            radios.Add("3ullA_1prtF", 5);
            radios.Add("3vub_2vubH", 3);
            radios.Add("4aahA_1flgB", 3);
            radios.Add("4bcl_1ksaA", 3);
            radios.Add("4crxA_1floA", 6);
            radios.Add("4icb_1mho", 3);
            radios.Add("4mbp_1eljA", 4);
            radios.Add("4mbp_1eu8A", 4);
            radios.Add("4mbp_1ezpA", 10);
            radios.Add("4mbp_1gggB", 3);
            radios.Add("4pgaA_1ho3A", 4);
            radios.Add("4uagA_1eehA", 4);
            radios.Add("5hpgA_2pk4", 8);
            radios.Add("5nul_1akt", 5);
            radios.Add("5nul_1j9gA", 3);
            radios.Add("5tmpA_3tmkG", 5);
            radios.Add("6prcL_6prcM", 4);
            radios.Add("7taa_2taaA", 5);
            radios.Add("8fabA_1tetH", 3);
            radios.Add("group1", 3);
            radios.Add("group2", 5);
            radios.Add("group3", 7);
            radios.Add("group4", 4);
            radios.Add("group5", 4);
            radios.Add("group6", 3);
            radios.Add("group7", 3);
            radios.Add("group8", 3);
            radios.Add("group9", 3);
            radios.Add("group10", 3);
            radios.Add("group11", 4);
            radios.Add("group12", 4);
            radios.Add("group13", 3);
            radios.Add("group14", 3);
            radios.Add("group15", 6);
            radios.Add("group16", 4);
            radios.Add("group17", 3);
            radios.Add("group18", 3);
            radios.Add("group19", 3);
            radios.Add("group20", 3);
            radios.Add("group21", 4);
            radios.Add("group22", 3);
            radios.Add("group23", 8);
            radios.Add("group24", 3);
            radios.Add("group25", 3);
            radios.Add("group26", 3);
            radios.Add("group27", 3);
            radios.Add("group28", 5);
            radios.Add("group29", 6);
            radios.Add("group30", 4);
            radios.Add("group31", 3);
            radios.Add("group32", 3);
            radios.Add("group33", 3);
            radios.Add("group34", 4);
            radios.Add("group35", 3);
            radios.Add("group36", 3);
            radios.Add("group37", 3);
            radios.Add("group38", 8);
            radios.Add("group39", 3);
            radios.Add("group40", 8);
            radios.Add("group41", 3);
            radios.Add("group42", 4);
            radios.Add("group43", 3);
            radios.Add("group44", 5);
            radios.Add("group45", 5);
            radios.Add("group46", 7);
            radios.Add("group47", 3);
            radios.Add("group48", 3);
            radios.Add("group49", 3);
            radios.Add("group50", 3);
            radios.Add("group51", 3);
            radios.Add("group52", 3);
            radios.Add("group53", 3);
            radios.Add("group54", 3);
            radios.Add("group55", 3);
            radios.Add("group56", 6);
            radios.Add("group57", 3);
            radios.Add("group58", 3);
            radios.Add("group59", 3);
            radios.Add("group60", 3);
            radios.Add("group61", 3);
            radios.Add("group62", 10);
            radios.Add("group63", 4);
            radios.Add("group64", 4);
            radios.Add("group65", 9);
            radios.Add("group66", 3);
            radios.Add("group67", 6);
            radios.Add("group68", 3);
            radios.Add("group69", 3);
            radios.Add("group70", 5);
            radios.Add("group71", 3);
            radios.Add("group72", 3);
            radios.Add("group73", 3);
            radios.Add("group74", 3);
            radios.Add("group75", 5);
            radios.Add("group76", 10);
            radios.Add("group77", 3);
            radios.Add("group78", 6);
            radios.Add("group79", 3);
            radios.Add("group80", 5);
            radios.Add("group81", 9);
            radios.Add("group82", 6);
            radios.Add("group83", 3);
            radios.Add("group84", 6);
            radios.Add("group85", 4);
            radios.Add("group86", 4);
            radios.Add("group87", 3);
            radios.Add("group88", 4);
            radios.Add("group89", 5);
            radios.Add("group90", 5);
            radios.Add("group91", 4);
            radios.Add("group92", 3);
            radios.Add("group93", 4);
            radios.Add("group94", 4);
            radios.Add("group95", 4);
            radios.Add("group96", 4);
            radios.Add("group97", 4);
            radios.Add("group98", 3);
            radios.Add("group99", 3);
            radios.Add("group100", 3);
            radios.Add("group101", 3);
            radios.Add("group102", 4);
            radios.Add("group103", 4);
            radios.Add("group104", 8);
            radios.Add("group105", 3);
            radios.Add("group106", 6);
            radios.Add("group107", 3);
            radios.Add("group108", 3);
            radios.Add("group109", 4);
            radios.Add("group110", 3);
            radios.Add("group111", 3);
            radios.Add("group112", 3);
            radios.Add("group113", 6);
            radios.Add("group114", 3);
            radios.Add("group115", 5);
            radios.Add("group116", 3);
            radios.Add("group117", 3);
            radios.Add("group118", 3);
            radios.Add("group119", 3);
            radios.Add("group120", 10);
            radios.Add("group121", 4);
            radios.Add("group122", 3);
            radios.Add("group123", 3);
            radios.Add("group124", 3);
            radios.Add("group125", 4);
            radios.Add("group126", 7);
            radios.Add("group127", 6);
            radios.Add("group128", 3);
            radios.Add("group129", 5);
            radios.Add("group130", 9);
            radios.Add("group131", 3);
            radios.Add("group132", 9);
            radios.Add("group133", 3);
            radios.Add("group134", 3);
            radios.Add("group135", 3);
            radios.Add("group136", 5);
            radios.Add("group137", 3);
            radios.Add("group138", 3);
            radios.Add("group139", 3);
            radios.Add("group140", 3);
            radios.Add("group141", 6);
            radios.Add("group142", 3);
            radios.Add("group143", 3);
            radios.Add("group144", 5);
            radios.Add("group145", 3);
            radios.Add("group146", 6);
            radios.Add("group147", 4);
            radios.Add("group148", 6);
            radios.Add("group149", 4);
            radios.Add("group150", 3);
            radios.Add("group151", 3);
            radios.Add("group152", 3);
            radios.Add("group153", 3);
            radios.Add("group154", 7);
            radios.Add("group155", 7);
            radios.Add("group156", 6);
            radios.Add("group157", 3);
            radios.Add("group158", 9);
            radios.Add("group159", 3);
            radios.Add("group160", 3);
            radios.Add("group161", 3);
            radios.Add("group162", 6);
            radios.Add("group163", 4);
            radios.Add("group164", 4);
            radios.Add("group165", 3);
            radios.Add("group166", 3);
            radios.Add("group167", 3);
            radios.Add("group168", 3);
            radios.Add("group169", 3);
            radios.Add("group170", 4);
            radios.Add("group171", 7);
            radios.Add("group172", 4);
            radios.Add("group173", 6);
            radios.Add("group174", 7);
            radios.Add("group175", 10);
            radios.Add("group176", 7);
            radios.Add("group177", 5);
            radios.Add("group178", 5);
            radios.Add("group179", 3);
            radios.Add("group180", 3);
            radios.Add("group181", 5);
            radios.Add("group182", 7);
            radios.Add("group183", 3);
            radios.Add("group184", 8);
            radios.Add("group185", 7);
            radios.Add("group186", 3);
            radios.Add("group187", 9);
            radios.Add("group188", 6);
            radios.Add("group189", 3);
            radios.Add("group190", 3);
            radios.Add("group191", 3);
            radios.Add("group192", 3);
            radios.Add("group193", 7);
            radios.Add("group194", 8);
            radios.Add("group195", 3);
            radios.Add("group196", 3);
            radios.Add("group197", 3);
            radios.Add("group198", 3);
            radios.Add("group199", 7);
            radios.Add("group200", 3);
            radios.Add("group201", 3);
            radios.Add("group202", 8);
            radios.Add("group203", 3);
            radios.Add("group204", 7);
            radios.Add("group205", 3);
            radios.Add("group206", 3);
            radios.Add("group207", 3);
            radios.Add("group208", 3);
            radios.Add("group209", 7);
            radios.Add("group210", 3);
            radios.Add("group211", 7);
            radios.Add("group212", 3);
            radios.Add("group213", 3);
            radios.Add("group214", 3);
            radios.Add("group215", 3);
            radios.Add("group216", 8);
            radios.Add("group217", 3);
            radios.Add("group218", 3);
            radios.Add("group219", 3);
            radios.Add("group220", 3);
            radios.Add("group221", 8);
            radios.Add("group222", 5);
            radios.Add("group223", 4);
            radios.Add("group224", 3);
            radios.Add("group225", 9);
            radios.Add("group226", 4);
            radios.Add("group227", 3);
            radios.Add("group228", 5);
            radios.Add("group229", 6);
            radios.Add("group230", 6);
            radios.Add("group231", 3);
            radios.Add("group232", 4);
            radios.Add("group233", 3);
            radios.Add("group234", 4);
            radios.Add("group235", 6);
            radios.Add("group236", 10);
            radios.Add("group237", 3);
            radios.Add("group238", 3);
            radios.Add("group239", 3);
            radios.Add("group240", 3);
            radios.Add("group241", 3);
            radios.Add("group242", 3);
            radios.Add("group243", 4);
            radios.Add("group244", 3);
            radios.Add("group245", 4);
            radios.Add("group246", 3);
            radios.Add("group247", 3);
            radios.Add("group248", 3);
            radios.Add("group249", 5);
            radios.Add("group250", 3);
            radios.Add("group251", 3);
            radios.Add("group252", 3);
            radios.Add("group253", 3);
            radios.Add("group254", 3);
            radios.Add("group255", 4);
            radios.Add("group256", 3);
            radios.Add("group257", 3);
            radios.Add("group258", 3);
            radios.Add("group259", 3);
            radios.Add("group260", 3);
            radios.Add("group261", 3);
            radios.Add("group262", 4);
            radios.Add("group263", 3);
            radios.Add("group264", 3);
            radios.Add("group265", 4);
            radios.Add("group266", 3);
            radios.Add("group267", 7);
            radios.Add("group268", 3);
            radios.Add("group269", 3);
            radios.Add("group270", 4);
            radios.Add("group271", 4);
            radios.Add("group272", 4);
            radios.Add("group273", 4);
            radios.Add("group274", 3);
            radios.Add("group275", 4);
            radios.Add("group276", 5);
            radios.Add("group277", 3);
            radios.Add("group278", 3);
            radios.Add("group279", 3);
            radios.Add("group280", 3);
            radios.Add("group281", 5);
            radios.Add("group282", 5);
            radios.Add("group283", 3);
            radios.Add("group284", 6);
            radios.Add("group285", 3);
            radios.Add("group286", 5);
            radios.Add("group287", 3);
            radios.Add("group288", 3);
            radios.Add("group289", 3);
            radios.Add("group290", 3);
            radios.Add("group291", 3);
            radios.Add("group292", 3);
            radios.Add("group293", 4);
            radios.Add("group294", 5);
            radios.Add("group295", 5);
            radios.Add("group296", 4);
            radios.Add("group297", 5);
            radios.Add("group298", 3);
            radios.Add("group299", 3);
            radios.Add("group300", 3);
            radios.Add("group301", 4);
            radios.Add("group302", 9);
            radios.Add("group303", 3);
            radios.Add("group304", 5);
            radios.Add("group305", 5);
            radios.Add("group306", 3);
            radios.Add("group307", 4);
            radios.Add("group308", 3);
            radios.Add("group309", 9);
            radios.Add("group310", 3);
            radios.Add("group311", 3);
            radios.Add("group312", 3);
            radios.Add("group313", 3);
            radios.Add("group314", 4);
            radios.Add("group315", 6);
            radios.Add("group316", 9);
            radios.Add("group317", 4);
            radios.Add("group318", 3);
            radios.Add("group319", 9);
            radios.Add("group320", 3);
            radios.Add("group321", 3);
            radios.Add("group322", 3);
            radios.Add("group323", 4);
            radios.Add("group324", 3);
            radios.Add("group325", 3);
            radios.Add("group326", 8);
            radios.Add("group327", 3);
            radios.Add("group328", 3);
            radios.Add("group329", 4);
            radios.Add("group330", 3);
            radios.Add("group331", 3);
            radios.Add("group332", 3);
            radios.Add("group333", 6);
            radios.Add("group334", 3);
            radios.Add("group335", 3);
            radios.Add("group336", 3);
            radios.Add("group337", 5);
            radios.Add("group338", 3);
            radios.Add("group339", 5);
            radios.Add("group340", 5);
            radios.Add("group341", 3);
            radios.Add("group342", 8);
            radios.Add("group343", 3);
            radios.Add("group344", 3);
            radios.Add("group345", 5);
            radios.Add("group346", 4);
            radios.Add("group347", 3);
            radios.Add("group348", 5);
            radios.Add("group349", 8);
            radios.Add("group350", 5);
            radios.Add("group351", 3);
            radios.Add("group352", 4);
            radios.Add("group353", 3);
            radios.Add("group354", 3);
            radios.Add("group355", 4);
            radios.Add("group356", 3);
            radios.Add("group357", 3);
            radios.Add("group358", 3);
            radios.Add("group359", 6);
            radios.Add("group360", 3);
            radios.Add("group361", 4);
            radios.Add("group362", 9);
            radios.Add("group363", 8);
            radios.Add("group364", 3);
            radios.Add("group365", 4);
            radios.Add("group366", 3);
            radios.Add("group367", 4);
            radios.Add("group368", 6);
            radios.Add("group369", 4);
            radios.Add("group370", 3);
            radios.Add("group371", 3);
            radios.Add("group372", 3);
            radios.Add("group373", 5);
            radios.Add("group374", 6);
            radios.Add("group375", 3);
            radios.Add("group376", 4);
            radios.Add("group377", 6);
            radios.Add("group378", 3);
            radios.Add("group379", 4);
            radios.Add("group380", 7);
            radios.Add("group381", 3);
            radios.Add("group382", 3);
            radios.Add("group383", 8);
            radios.Add("group384", 3);
            radios.Add("group385", 3);
            radios.Add("group386", 5);
            radios.Add("group387", 3);
            radios.Add("group388", 3);
            radios.Add("group389", 6);
            radios.Add("group390", 3);
            radios.Add("group391", 3);
            radios.Add("group392", 4);
            radios.Add("group393", 3);
            radios.Add("group394", 10);
            radios.Add("group395", 3);
            radios.Add("group396", 5);
            radios.Add("group397", 3);
            radios.Add("group398", 3);
            radios.Add("group399", 3);
            radios.Add("group400", 5);
            radios.Add("group401", 4);
            radios.Add("group402", 9);
            radios.Add("group403", 4);
            radios.Add("group404", 9);
            radios.Add("group405", 4);
            radios.Add("group406", 3);
            radios.Add("group407", 7);
            radios.Add("group408", 3);
            radios.Add("group409", 6);
            radios.Add("group410", 3);
            radios.Add("group411", 3);
            radios.Add("group412", 6);
            radios.Add("group413", 4);
            radios.Add("group414", 4);
            radios.Add("group415", 3);
            radios.Add("group416", 4);
            radios.Add("group417", 3);
            radios.Add("group418", 5);
            radios.Add("group419", 3);
            radios.Add("group420", 4);
            radios.Add("group421", 3);
            radios.Add("group422", 4);
            radios.Add("group423", 3);
            radios.Add("group424", 3);
            radios.Add("group425", 3);

            return radios;
        }

        private static int Main(string[] args)
        {
            //analizer(args);
            compare(args);
            return 0;
        }

        private static void analizer(string[] args)
        {
            Dictionary<string, int> radios = new Dictionary<string, int>();
            radios = fill();
            string path2 = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().GetName().CodeBase.Substring(8));

            string tofile = "";
            string header = "filename\tNumber of Sequences\tAverage length\tStandar Deviation\tAverage Dist\tSD Dist";
            //Console.WriteLine(tofile);
            for (int c = 0; c != args.Count(); c++)
            {
                string file = "";
                file = Path.GetFileNameWithoutExtension(args[c]);
                if (file == "group")
                {
                    file = args[c].Substring(0, args[c].LastIndexOf('\\'));
                    file = file.Substring(file.LastIndexOf('\\'), file.Length - file.LastIndexOf('\\'));
                    file = file.Substring(1, file.Length - 1);
                }

                Sequencer all = new Sequencer(args[c], true);
                Aligner a = new Aligner();
                double[] avDist;
                a.gap(-2, -4, -3);

                avDist = a.averageDistance(all, "propossal", radios[file], 12);
                if (radios.ContainsKey(file))
                {
                    tofile += file + "\t" + all.count() + "\t" + all.meanLength() + "\t" + all.stdDeviation() + "\t" + avDist[0] + "\t" + avDist[1] + "\n";
                    //tofile += file + "\t" + all.count() + "\t" + all.meanLength() + "\t" + all.stdDeviation() + "\n";
                    if (!File.Exists(path2 + "\\ResultsAn.txt"))
                    {
                        StreamWriter res = new StreamWriter(path2 + "\\ResultsAn.txt", true);
                        res.Write(header + "\n");
                        res.Write(tofile);
                        //Console.Write(header + "\n");
                        //Console.Write(tofile);
                        tofile = "";
                        res.Close();
                    }
                    else
                    {
                        StreamWriter res = new StreamWriter(path2 + "\\ResultsAn.txt", true);
                        res.Write(tofile);
                        //Console.Write(tofile);
                        tofile = "";
                        res.Close();
                    }
                }
            }
        }

        private static void Help()
        {
            
            Console.WriteLine("     ┌───────────────────────────────────────────────────────────────────┐     ");
            Console.WriteLine("     │   Sequence Alignment by Radial Evaluation of Local Interactions   │     ");
            Console.WriteLine("     │                             SARELI                                │     ");
            Console.WriteLine("     ├───────────────────────────────────────────────────────────────────┤     ");
            Console.WriteLine("     │                  Author: McS Ricardo Ortega Magaña                │     ");
            Console.WriteLine("     ├───────────────────────────────────────────────────────────────────┤     ");
            Console.WriteLine("     │                  Advisor: Dr. Arturo Chavoya Peña                 │     ");
            Console.WriteLine("     ├───────────────────────────────────────────────────────────────────┤     ");
            Console.WriteLine("     │                  MIT License, Copyright (c) 2016                  │     ");
            Console.WriteLine("     │                  Version:        1.11.00002                       │     ");
            Console.WriteLine("     └───────────────────────────────────────────────────────────────────┘\n\n");
            Console.WriteLine("  ┌──────────────────────────────────────────────────────────────────────────┐");
            Console.WriteLine("  │    Welcome to the multiple sequence alignment software implementation    │");
            Console.WriteLine("  │  for proteins developed in the department of information systems,        │");
            Console.WriteLine("  │  located at the Centro Universitario de Ciencias Económico               │");
            Console.WriteLine("  │  Administrativas, Universidad de Guadalajara, Zapopan, Jalisco, México.  │");
            Console.WriteLine("  │                                                                          │");
            Console.WriteLine("  └──────────────────────────────────────────────────────────────────────────┘\n\n");
            Console.WriteLine("  ┌──────────────────────────────────────────────────────────────────────────┐");
            Console.WriteLine("  │                             INTRODUCTION                                 │");
            Console.WriteLine("  ├──────────────────────────────────────────────────────────────────────────┤");
            Console.WriteLine("  │     This command line utility performs a number of multiple sequence     │");
            Console.WriteLine("  │     tasks for proteins, such as alignment, scoring, and generation and   │");
            Console.WriteLine("  │     COMPARISON OF STATISTICS OF INPUT SEQUENCE FILES. THE INPUT FILES    │".ToLower());
            Console.WriteLine("  │     CAN BE IN ANY OF THE SUPPORTED FILES FORMATS                         │".ToLower());
            Console.WriteLine("  │     (MSF, ALN, PAMSA, FAS, TFA, FASTA).                                  │".ToLower());
            Console.WriteLine("  │                                                                          │");
            Console.WriteLine("  │       For a description of all the options, see the man page section     │");
            Console.WriteLine("  │     BELOW.                                                               │".ToLower());
            Console.WriteLine("  │                                                                          │");
            Console.WriteLine("  │       The default output is a file in fasta format with the alignment    │");
            Console.WriteLine("  │     CORRESPONDING TO THE HIGHEST COLUMN SCORE FOUND. THE NAME OF THE     │".ToLower());
            Console.WriteLine("  │     OUTPUT FILE IS AUTOMATICALLY GENERATED STARTING WITH THE PREFIX      │".ToLower());
            Console.WriteLine("  │     “SARELI_MaxCS_”, FOLLOWED BY \"R\" AND THE BEST RADIUS FOUND FOR       │".ToLower());
            Console.WriteLine("  │     THE RANGE PROVIDED, AND ENDING WITH THE NAME OF THE ORIGINAL FILE    │".ToLower());
            Console.WriteLine("  │     WITH THE FASTA EXTENSION                                             │".ToLower());
            Console.WriteLine("  │     (e.g. Sareli_MaxCS_R3_originalfilename.fasta).                       │".ToLower());
            Console.WriteLine("  │                                                                          │");
            Console.WriteLine("  │        Optionally, a file with the highest sum of pairs score can also   │");
            Console.WriteLine("  │     BE OUTPUT BY SPECIFYING THE -sp FLAG. IN BOTH CASES, A FILE WITH     │".ToLower());
            Console.WriteLine("  │     SCORES CAN BE OUTPUT WHEN THE -s OPTION IS SPECIFIED.                │".ToLower());
            Console.WriteLine("  └──────────────────────────────────────────────────────────────────────────┘\n\n");
            Console.WriteLine("  ┌──────────────────────────────────────────────────────────────────────────┐");
            Console.WriteLine("  │                                Man page                                  │");
            Console.WriteLine("  ├──────────────────────────────────────────────────────────────────────────┤");
            Console.WriteLine("  │       SYNTAX:                                                            │");
            Console.WriteLine("  │                                                                          │");
            Console.WriteLine("  │              SARELI -in FILENAME [options]                               │");
            Console.WriteLine("  │                                                                          │");
            Console.WriteLine("  │       OPTIONS:                                                           │");
            Console.WriteLine("  │                                                                          │");
            Console.WriteLine("  │              -t                                                          │");
            Console.WriteLine("  │                   Evaluate an alignment with the sum of pairs and        │");
            Console.WriteLine("  │                   THE COLUMN SCORE. THIS PARAMETER DISABLES THE          │".ToLower());
            Console.WriteLine("  │                   ALIGNMENT TOOLS.                                       │".ToLower());
            Console.WriteLine("  │                                                                          │");
            Console.WriteLine("  │              -s FILENAME                                                 │");
            Console.WriteLine("  │                   Name of the file to output the results in a CSV        │");
            Console.WriteLine("  │                   (Comma-Separated Values) format. If not provided,      │");
            Console.WriteLine("  │                   STDOUT IS USED.                                        │".ToLower());
            Console.WriteLine("  │                                                                          │");
            Console.WriteLine("  │              -r XX,YY                                                    │");
            Console.WriteLine("  │                   This parameter searches from radius value XX to YY     │");
            Console.WriteLine("  │                   (With YY>XX and XX>0) for the best alignment.          │");
            Console.WriteLine("  │                   (Defaults to 3 to 10 if not provided).                 │");
            Console.WriteLine("  │                                                                          │");
            Console.WriteLine("  │              -r1 .XX                                                     │");
            Console.WriteLine("  │                   Threshold value for the first refinement method        │");
            Console.WriteLine("  │                   (Defaults to .75 if not provided).                     │");
            Console.WriteLine("  │                                                                          │");
            Console.WriteLine("  │              -r2 .XX                                                     │");
            Console.WriteLine("  │                   Threshold value for the second refinement method       │");
            Console.WriteLine("  │                   (Defaults to .40 if not provided).                     │");
            Console.WriteLine("  │                                                                          │");
            Console.WriteLine("  │              -in FILENAME                                                │");
            Console.WriteLine("  │                   Path for the file with the protein sequences to align. │");
            Console.WriteLine("  │                                                                          │");
            Console.WriteLine("  │              -sp                                                         │");
            Console.WriteLine("  │                   Additionally to the default output file with the       │");
            Console.WriteLine("  │                   HIGHEST COLUMN SCORE, A FILE WITH THE HIGHEST          │".ToLower());
            Console.WriteLine("  │                   SUM OF PAIRS SCORE CAN ALSO BE OUTPUT, CONTAINING      │".ToLower());
            Console.WriteLine("  │                   \"MaxSP\" IN THE FILENAME GENERATED, INSTEAD OF \"MaxC\".  │".ToLower());
            Console.WriteLine("  │                   (Recommended)                                          │");
            Console.WriteLine("  │                                                                          │");
            Console.WriteLine("  │              -ptx cudaKernel                                             │");
            Console.WriteLine("  │                   Path for the CUDA kernel file to use as accelerator.   │");
            Console.WriteLine("  │                                                                          │");
            Console.WriteLine("  │              -i XX                                                       │");
            Console.WriteLine("  │                   Maximum number of iterations for the first             │");
            Console.WriteLine("  │                   refinement method (Defaults to 5 if not provided).     │");
            Console.WriteLine("  └──────────────────────────────────────────────────────────────────────────┘\n\n");
            Console.ReadKey();
        }

        private static int compare(string[] args)
        {
            string header = "";
            String tofile = "";
            int itera = 5;

            //tofile = "File\tBlosum62\tSimple\n";
            Stopwatch timer = new Stopwatch();

            string file = "";
            string fileName = "";

            //Console.WriteLine(args[c] + " " + c + "/" + args.Length);
            //if (Path.GetExtension(args[c]).ToUpper() == ".FASTA" && !File.Exists("muscle_" + file + ".fasta"))
            //if (Path.GetExtension(args[c]).ToUpper() == ".FASTA" || Path.GetExtension(args[c]).ToUpper() == ".TFA")
            // if (!File.Exists("muscle_" + file + ".fasta"))
            List<string> arg = new List<string>();
            string rFile = ""; //Output file (-s option)
            double r1 = 0;   //First refinement threshold (-r1 option)
            double r2 = 0;   //First refinement threshold (-r1 option)
            bool sp = false;
            bool useCuda = false;
            bool test = false;
            string cudaKernelFileName = "";
            arg = args.ToList<string>();
            for (int c = 0; c != arg.Count; c++) {
                arg[c] = arg[c].ToLower();
            }

                if (arg.Contains("-ptx"))
                {

                    try
                    {
                        cudaKernelFileName = arg[arg.IndexOf("-ptx") + 1];
                        useCuda = true;
                    }
                    catch (Exception E)
                    {
                        Help();
                        return 0;
                    }
                }
            int firstRadious = 3;
            int lastRadious = 10;

            if (arg.Contains("-r"))
            {
                try
                {
                    firstRadious = Convert.ToInt32(arg[arg.IndexOf("-r") + 1].Split(',')[0]);
                    if (firstRadious < 1) {
                        Help();
                        return 0;
                    }
                    lastRadious = Convert.ToInt32(arg[arg.IndexOf("-r") + 1].Split(',')[1]);
                }
                catch (Exception E)
                {
                    Help();
                    return 0;
                }
            }
            if (arg.Contains("-sp"))
            {
                sp = true;
            }

            if (arg.Contains("-in"))
            {
                try
                {
                    fileName = arg[arg.IndexOf("-in") + 1];
                }
                catch (Exception E)
                {
                    Help();
                    return 0;
                }
            }

            if (arg.Contains("-t"))
            {
                   test = true;
            }

            if (arg.Contains("-i"))
            {
                try
                {
                    itera = Convert.ToInt32(arg[arg.IndexOf("-i") + 1]);
                }
                catch (Exception E)
                {
                    Help();
                    return 0;
                }
            }

            try
            {
                file = Path.GetFileNameWithoutExtension(fileName);
            }
            catch (Exception E)
            {
                Help();
                return 1;
            }

            if (arg.Contains("-s"))
            {
                try
                {
                    rFile = arg[arg.IndexOf("-s") + 1];
                }
                catch (Exception E)
                {
                    Help();
                    return 0;
                }
            }

            if (arg.Contains("-r1"))
            {
                try
                {
                    r1 = Double.Parse(arg[arg.IndexOf("-r1") + 1]);
                }
                catch (Exception E)
                {
                    Help();
                    return 0;
                }
            }
            if (arg.Contains("-r2"))
            {
                try
                {
                    r2 = Double.Parse(arg[arg.IndexOf("-r2") + 1]);
                }
                catch (Exception E)
                {
                    Help();
                    return 0;
                }
            }

            if (fileName == "")
            {
                Help();
                return 0;
            }

            if (!test)
            {
                header = "";

                double nowScore = 0;
                Sequencer all = new Sequencer(fileName, true);

                string path = Path.GetDirectoryName(fileName);
                Aligner worker = new Aligner(all);

                Sequencer[] NHT_2 = new Sequencer[lastRadious - firstRadious];
                Stopwatch a = new Stopwatch();

                header += "File,";
                tofile += file + ",";
                int maxRefCS = 0;
                int maxRefSP = 0;
                int maxRSP = 0;
                int maxRCS = 0;
                int maxRSP_cs = 0;
                int maxRCS_sp = 0;

                string path2 = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().GetName().CodeBase.Substring(8));
                Dictionary<string, int> radios = new Dictionary<string, int>();
                List<string> skipping = new List<string>();

                double refi = r1;

                for (int k = firstRadious; k != lastRadious; k++)
                {
                    Console.WriteLine("Aligning with R = " + k);

                    worker.gap(-2, -5, -4);
                    if (useCuda)
                    {
                        a.Reset();
                        a.Start();
                        int n = 24;
                        bool po = true;
                        while (po && n > 0)
                        {
                            try
                            {
                                NHT_2[k - firstRadious] = worker.alignByNeighbourJoiningCUDA(new Sequencer(fileName, true), "propossal", k, n);
                                po = false;
                            }
                            catch (Exception E)
                            {
                                Console.WriteLine("Too many sequences aligning at once, next try using " + (n * n) + " threads");
                            }
                            n -= 6;
                        }
                        if (n < 1)
                        {
                            Console.WriteLine("This sequence file cannot be processed in this graphic card with this software");
                            return 0;
                        }

                        a.Stop();
                        header += "Time radio:" + k + ",";
                        tofile += a.ElapsedMilliseconds + ",";

                        worker = new Aligner(NHT_2[k - firstRadious]);

                        // nowScore = worker.sumOfPairs();
                        // header += "SP radio:" + k + ",";
                        //  tofile += String.Format("{0:0}", nowScore) + ",";

                        //  nowScore = worker.columnScore(NHT_2[k - firstRadious]);
                        //  header += "CS radio:" + k + ",";

                        //   tofile += String.Format("{0:0}", nowScore) + ",";
                        Sequencer temp = worker.refineAlign(new Sequencer(NHT_2[k - firstRadious].getFasta(), false), refi, itera, useCuda);

                        NHT_2[k - 3] = worker.refineAlign2(temp, r2);
                        header += "SP radio:" + k + ",";
                        nowScore = new Aligner(NHT_2[k - firstRadious]).sumOfPairs();
                        int t = (int)nowScore;
                        tofile += String.Format("{0:0}", nowScore) + ",";
                        if (nowScore > maxRefSP || k == firstRadious)
                        {
                            maxRefSP = (int)nowScore;
                            maxRSP = k;
                            maxRSP_cs = (int)worker.columnScore(NHT_2[k - firstRadious]);
                        }
                        nowScore = worker.columnScore(NHT_2[k - firstRadious]);
                        if (nowScore > maxRefCS || k == firstRadious)
                        {
                            maxRefCS = (int)nowScore;
                            maxRCS_sp = t;
                            maxRCS = k;
                        }

                        header += "CS radio:" + k + ",";
                        tofile += String.Format("{0:0.00}", nowScore) + ",";
                    }
                    else
                    {
                        a.Reset();
                        a.Start();
                        NHT_2[k - firstRadious] = worker.alignByNeighbourJoining(all, k);
                        a.Stop();
                        header += "Time radio:" + k + ",";
                        tofile += a.ElapsedMilliseconds + ",";

                        worker = new Aligner(NHT_2[k - firstRadious]);
                        // nowScore = worker.sumOfPairs();
                        // header += "SP radio:" + k + ",";
                        //tofile += String.Format("{0:0}", nowScore) + ",";

                        //  nowScore = worker.columnScore(NHT_2[k - firstRadious]);
                        // header += "CS radio:" + k + ",";

                        // tofile += String.Format("{0:0}", nowScore) + ",";
                        Sequencer temp = worker.refineAlign(new Sequencer(NHT_2[k - firstRadious].getFasta(), false), refi, itera, useCuda);

                        NHT_2[k - firstRadious] = worker.refineAlign2(temp, r2);
                        header += "SP radio:" + k + ",";
                        nowScore = new Aligner(NHT_2[k - firstRadious]).sumOfPairs();
                        int t = (int)nowScore;
                        tofile += String.Format("{0:0}", nowScore) + ",";
                        if (nowScore > maxRefSP || k == firstRadious)
                        {
                            maxRefSP = (int)nowScore;
                            maxRSP = k;
                            maxRSP_cs = (int)worker.columnScore(NHT_2[k - firstRadious]);
                        }
                        nowScore = worker.columnScore(NHT_2[k - firstRadious]);
                        if (nowScore > maxRefCS || k == firstRadious)
                        {
                            maxRefCS = (int)nowScore;
                            maxRCS_sp = t;
                            maxRCS = k;
                        }

                        header += "CS radio:" + k + ",";
                        tofile += String.Format("{0:0.00}", nowScore) + ",";
                    }
                }
                string tech = "";
                if (useCuda)
                {
                    tech = "cuda";
                }
                else
                {
                    tech = "ser";
                }
                NHT_2[maxRCS - firstRadious].print(0, -1, false, "Sareli_MaxCS_R" + maxRCS + "_" + file + ".fasta");
                if (sp) {
                    NHT_2[maxRSP - firstRadious].print(0, -1, false, "Sareli_MaxSP_R" + maxRSP + "_" + file + ".fasta");
                }
                tofile = tofile.Substring(0, tofile.Length - 1);
                if (rFile.Length > 0)
                    if (!File.Exists(path2 + "\\" + rFile))
                    {
                        StreamWriter res = new StreamWriter(path2 + "\\" + rFile, true);
                        res.Write(header + "type,Date\n");
                        res.Write(tofile + "," + tech + "," + DateTime.Today + "\n");
                        res.Close();
                    }
                    else
                    {
                        StreamWriter res = new StreamWriter(path2 + "\\" + rFile, true);
                        res.Write(tofile + "," + tech + "," + DateTime.Today + "\n");
                        tofile = "";
                        res.Close();
                    }
            }
            else {
                Aligner worker = new Aligner(new Sequencer(fileName, true));
                string path2 = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().GetName().CodeBase.Substring(8));
                header += "File,";
                tofile += file + ",";
                
                header += "SP,";
                int nowScore = worker.sumOfPairs();
                tofile += String.Format("{0:0}", nowScore) + ",";
                nowScore = (int)worker.columnScore(new Sequencer(fileName, true));
                header += "CS,";
                tofile += String.Format("{0:0}", nowScore) + ",";
                tofile = tofile.Substring(0, tofile.Length - 1);
                header = header.Substring(0, header.Length - 1);
                if (rFile.Length > 0)
                {
                    //Console.WriteLine(header);
                    //Console.WriteLine(tofile);
                    if (!File.Exists(path2 + "\\" + rFile))
                    {
                        StreamWriter res = new StreamWriter(path2 + "\\" + rFile, true);
                        res.Write(header + ",Date\n");
                        res.Write(tofile + "," + DateTime.Today + "\n");
                        res.Close();
                    }
                    else
                    {
                        StreamWriter res = new StreamWriter(path2 + "\\" + rFile, true);
                        res.Write(tofile + "," + DateTime.Today + "\n");
                        res.Close();
                    }
                }
                else {
                    Console.WriteLine(header);
                    Console.WriteLine(tofile);
                }
            }
            return 0;
        }
    }
}

/*

            Sequencer Clustalw = new Sequencer(args[c].Substring(0, args[c].LastIndexOf('.')) + ".aln", true);
            Aligner clusWorker = new Aligner(Clustalw);
            header += "SP clustalw2,Time ClustalW2,";
            nowScore = clusWorker.sumOfPairs();
            tofile += String.Format("{0:0}", nowScore) + "," + a.ElapsedMilliseconds + ",";

            header += "CS clustalw2,";
            nowScore = worker.columnScore(new Sequencer(args[c].Substring(0, args[c].LastIndexOf('.')) + ".aln", true));
            tofile += String.Format("{0:0}", nowScore) + ",";

            a.Reset();
            a.Start();
            command = new Process();
            command.EnableRaisingEvents = false;
            command.StartInfo.FileName = path2 + "\\clustalo.exe";
            command.StartInfo.Arguments = "-i \"" + args[c] + "\" -o clustalo_" + file + ".fasta";
            command.StartInfo.UseShellExecute = false;
            command.StartInfo.RedirectStandardOutput = true;
            command.Start();
            while (!command.StandardOutput.EndOfStream)
            {
                command.StandardOutput.ReadLine();
            }
            while (!command.HasExited) ;

            a.Stop();
            header += "SP Clustalo,";
            nowScore = new Aligner(new Sequencer("clustalo_" + file + ".fasta", true)).sumOfPairs();
            tofile += String.Format("{0:0}", nowScore) + ",";
            header += "CS clustalo,";
            nowScore = worker.columnScore(new Sequencer("clustalo_" + file + ".fasta", true));
            tofile += String.Format("{0:0}", nowScore) + ",";

            a.Reset();
            a.Start();
            command = new Process();
            command.EnableRaisingEvents = false;

            command.StartInfo.FileName = path2+"\\mafft.bat";
            command.StartInfo.Arguments = "--maxiterate 1000 --globalpair " + args[c] + " > mafft_" + file + ".fasta";
            command.StartInfo.UseShellExecute = false;
            command.StartInfo.RedirectStandardOutput = true;
            command.Start();
            while (!command.StandardOutput.EndOfStream)
            {
                command.StandardOutput.ReadLine();
            }
            while (!command.HasExited) ;

            a.Stop();

            Clustalw = new Sequencer("mafft_" + file + ".fasta", true);
            clusWorker = new Aligner(Clustalw);
            header += "SP mafft,";
            tofile += clusWorker.sumOfPairs() + ",";

            header += "CS Mafft,";
            nowScore = worker.columnScore(new Sequencer("mafft_" + file + ".fasta", true));
            tofile += String.Format("{0:0}", nowScore) + ",";

            a.Restart();
            command = new Process();
            command.EnableRaisingEvents = false;
            command.StartInfo.FileName = "muscle.exe";
            command.StartInfo.Arguments = "-in \"" + args[c] + "\" -out muscle_" + file + ".fasta";
            command.StartInfo.UseShellExecute = false;
            command.StartInfo.RedirectStandardOutput = true;
            command.Start();
            while (!command.StandardOutput.EndOfStream)
            {
                command.StandardOutput.ReadLine();
            }
            while (!command.HasExited) ;
            a.Stop();

            Clustalw = new Sequencer("muscle_" + file + ".fasta", true);
            clusWorker = new Aligner(Clustalw);
            header += "SP Muscle, Time Muscle (ms),";
            tofile += clusWorker.sumOfPairs() + "," + a.ElapsedMilliseconds + ",";

            header += "CS Muscle,";
            nowScore = worker.columnScore(new Sequencer("muscle_" + file + ".fasta", true));
            tofile += String.Format("{0:0}", nowScore) + ",";
            */