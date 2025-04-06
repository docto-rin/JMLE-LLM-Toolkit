# -*- coding: utf-8 -*-
"""JMLE DPO Dataset Generator - Mistake Simulation Strategy (Refactored & Tag Adjusted)

高品質SFTデータを 'chosen' とし、LLMに特定の誤答を選択させ、
その理由付けを生成させて 'rejected' とするDPOデータセットを作成する。
タグ形式を <think>...</think>\nanswer:...\nexplanation:... に調整。
"""

import os
import json
import time
import random
import argparse
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login as hf_login, HfApi
from datasets import load_dataset, Dataset, DatasetDict, Features, Value
import re
from openai import OpenAI
import numpy as np

### --- 環境変数とログイン --- ###
# (変更なし)
load_dotenv(override=True)
hf_token = os.getenv('HF_TOKEN')
gemini_api_key = os.getenv('GEMINI_API_KEY')

if not gemini_api_key:
    raise ValueError("環境変数 GEMINI_API_KEY が設定されていません。.envファイルを確認してください。")

hf_logged_in = False
if hf_token:
    try:
        hf_login(token=hf_token)
        hf_logged_in = True
        print("Hugging Faceにログインしました。")
    except Exception as e:
        print(f"Hugging Faceログインエラー: {e}. プライベートデータセットの読み込みやアップロードに失敗する可能性があります。")
else:
    print("HF_TOKEN が設定されていません。プライベートデータセットの読み込みやHubへのアップロードはできません。")


### --- OpenAIクライアント設定 --- ###
# (変更なし)
def create_client(api_key, base_url):
    return OpenAI(api_key=api_key, base_url=base_url)

### --- プロンプト生成関数 --- ###

# --- [修正箇所 1] create_system_prompt の出力例変更 ---
def create_system_prompt():
    """システムプロンプト (タグ形式調整)"""
    return (
        "あなたは医師国家試験の過去問を解くにあたり、"
        "医学を隅々まで理解している非常に優秀で論理的なアシスタントです。\n\n"
        "【重要ルール】\n"
        "1. <think> ～ </think> の間に、論理的推論プロセスをステップごとに書いてください。\n"
        "2. </think> タグの後に改行し、次の形式で必ず出力してください:\n" # <o>タグ削除
        "   answer: [最終的な答え]\n"
        "   explanation: [医学的な根拠や理由]\n\n"
        "3. 問題文に「2つ選べ」などの明確な記載がある場合は、answer に複数の選択肢をカンマ区切り(a, bなど)で書いてください。\n"
        "4. 五者択一の場合はanswerを1つ(a, b, c, d, eのいずれか)にしてください。\n"
        "5. 数値入力問題の場合はanswerを求める数値だけにしてください（小数点以下の四捨五入などは問題文の指示に従う）。\n"
        "6. タグは必ず正しく閉じ、Markdown記法や余計な装飾はしないでください。"
    )

# (変更なし)
def format_choices(choices):
    if not choices: return "（本問は選択肢のない問題形式です）"
    return '\n'.join([str(c) for c in choices if c is not None])

# (変更なし)
def format_answer(answer):
    if isinstance(answer, list): return ', '.join(map(str, [a for a in answer if a is not None]))
    return str(answer) if answer is not None else ""

# (変更なし)
def extract_choice_label(choice_str):
    match = re.match(r"([a-zA-Z])[\.\s]", str(choice_str)) # str()で囲んで安全に
    return match.group(1) if match else None

# --- [修正箇所 2] create_user_prompt の出力例変更 ---
def create_user_prompt(example: dict) -> str:
    """思考を促すユーザープロンプト (タグ形式調整) - DPOの 'prompt' カラム用"""
    choices_str = format_choices(example.get('choices'))
    choice_part = f"【選択肢】\n{choices_str}\n\n" if example.get('choices') else ""
    return (
        "以下の問題文・選択肢を踏まえ、最終解答に至るまでの詳細な思考過程と、最終解答を指定の形式で出力してください。\n\n" # 「指定のタグで」を「指定の形式で」に変更
        f"【問題番号】{example.get('id', 'N/A')}\n"
        f"【問題文】\n{example.get('question', '')}\n\n"
        f"{choice_part}"
        "【出力例】\n"
        "<think>\n...\n</think>\n" # <o>タグ削除
        "answer: ...\n"
        "explanation: ...\n\n"
        "それでは、上記ルールに従って出力してください。"
    )

# --- [修正箇所 3] create_unified_prompt の呼び出し先変更 ---
# この関数自体は変更不要だが、呼び出す create_system_prompt と create_user_prompt が変更されている
def create_unified_prompt(example: dict) -> str:
    system_part = create_system_prompt()
    user_part = create_user_prompt(example)
    problem_info = user_part
    return f"{system_part}\n\n{problem_info}"

# --- [修正箇所 4] create_rejected_prompt の出力形式例変更 ---
def create_rejected_prompt(example: dict) -> tuple[str | None, str | None]:
    """
    指定された間違った選択肢を選ぶように指示するプロンプトを生成する (タグ形式調整)。
    成功した場合、(生成用プロンプト, 選択された誤答ラベル) を返す。
    誤答選択肢を選べない場合、(None, None) を返す。
    """
    choices = example.get('choices')
    correct_answers_raw = example.get('answer')
    question = example.get('question', '')
    example_id = example.get('id', 'N/A')

    if not choices or not isinstance(choices, list) or len(choices) == 0:
        return None, None

    correct_answers = []
    if isinstance(correct_answers_raw, list):
        correct_answers = [str(a).strip() for a in correct_answers_raw if a is not None]
    elif correct_answers_raw is not None:
        correct_answers = [str(correct_answers_raw).strip()]

    if not correct_answers:
         return None, None

    incorrect_choice_labels = []
    all_choice_labels = []
    for choice_str in choices:
        if choice_str is None: continue
        label = extract_choice_label(str(choice_str))
        if label:
            all_choice_labels.append(label)
            if label not in correct_answers:
                incorrect_choice_labels.append(label)

    if not incorrect_choice_labels:
        return None, None

    target_rejected_choice = random.choice(incorrect_choice_labels)

    system_prompt = create_system_prompt()
    choices_formatted = format_choices(choices)
    choice_part = f"【選択肢】\n{choices_formatted}\n\n"

    user_prompt_content = (
        f"あなたはAIアシスタントです。これから医師国家試験の問題に対する「望ましくない回答」(rejected)をDPOデータセット用に生成します。\n\n"
        f"以下の問題と選択肢について、**必ず選択肢 {target_rejected_choice} が正解であるという結論に至るように**、思考プロセス（<think>タグ内）と最終的な回答を生成してください。\n\n" # タグ名変更
        f"生成する思考プロセスは、一見論理的に見えるかもしれませんが、**選択肢 {target_rejected_choice} を選ぶために、医学的・倫理的な誤り、重要な情報の無視、論理の飛躍、不適切な優先順位付けなどを意図的に含めてください。**\n\n"
        f"**重要：最終的な answer は必ず {target_rejected_choice} としてください。**\n\n"
        f"【問題番号】{example_id}\n"
        f"【問題文】\n{question}\n\n"
        f"{choice_part}"
        "【出力形式】\n"
        "<think>\nここに、選択肢 {target_rejected_choice} を正当化するための、意図的に誤りを含む思考プロセスを記述してください。\n</think>\n" # タグ名変更、<o>削除
        "answer: {target_rejected_choice}\n"
        "explanation: ここに、選択肢 {target_rejected_choice} が正しいとする理由（誤りを含む）を記述してください。\n"
    )

    full_prompt_for_api = f"{system_prompt}\n\n{user_prompt_content}"
    return full_prompt_for_api, target_rejected_choice


### --- Gemini API呼び出し関数 --- ###
# (変更なし)
def call_gemini_for_rejected(
    prompt: str, client, model_name, system_prompt=None, temperature=0.7, top_p=0.95, retry_count=0
) -> str | None:
    messages = [{"role": "user", "content": prompt}]
    try:
        response = client.chat.completions.create(
             model=model_name,
             messages=messages,
             temperature=temperature,
             top_p=top_p,
        )
        if response and response.choices: return response.choices[0].message.content
        else: print(f"警告: APIから空のレスポンス Prompt: {prompt[:100]}..."); return None
    except Exception as e:
        if ("rate limit" in str(e).lower() or "quota" in str(e).lower() or "503" in str(e) or "500" in str(e)) and retry_count < 5:
             wait_time = (2 ** retry_count) * 10 + random.uniform(0, 5)
             print(f"APIエラー(リトライ可能): {str(e)}, {wait_time:.1f}秒後リトライ ({retry_count+1}/5)..."); time.sleep(wait_time)
             return call_gemini_for_rejected(prompt, client, model_name, system_prompt, temperature, top_p, retry_count + 1)
        elif retry_count < 3:
            wait_time = (2 ** retry_count) * 5 + random.uniform(0, 1)
            print(f"APIエラー: {str(e)}, {wait_time:.1f}秒後リトライ ({retry_count+1}/3)..."); time.sleep(wait_time)
            return call_gemini_for_rejected(prompt, client, model_name, system_prompt, temperature, top_p, retry_count + 1)
        else: print(f"最大リトライ回数到達 エラー: {str(e)} Prompt: {prompt[:100]}..."); return None


### --- 応答抽出関数 --- ###
# --- [修正箇所 5] extract_content_for_dpo の抽出ロジックと整形を変更 ---
def extract_content_for_dpo(response: str | None):
    """応答文字列から <think> 内容、answer、explanation を抽出する (タグ形式調整)"""
    if response is None: return None, False, None

    # 1. <think> タグの内容を抽出
    thoughts_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
    thoughts = thoughts_match.group(1).strip() if thoughts_match else ""

    # 2. <think> タグの後の部分から answer と explanation を抽出
    #    </think> の後から検索を開始する
    content_after_think = ""
    if thoughts_match:
        content_after_think = response[thoughts_match.end():].strip()
    else:
        # <think> タグが見つからない場合、応答全体から探す（フォールバック）
        content_after_think = response.strip()

    # 3. answer を抽出 (より頑健なパターン)
    #    行頭の "answer:" にマッチし、次の行が始まるまで（または文字列の終わりまで）取得
    answer_match = re.search(r"^\s*answer:\s*(.*?)(?=\n\s*\S+:|\Z)", content_after_think, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    extracted_answer = answer_match.group(1).strip() if answer_match else ""

    # 4. explanation を抽出 (より頑健なパターン)
    #    行頭の "explanation:" にマッチし、文字列の終わりまで取得（複数行対応）
    explanation_match = re.search(r"^\s*explanation:\s*(.*)", content_after_think, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    explanation = explanation_match.group(1).strip() if explanation_match else ""

    # 5. 成功判定: <think> タグと answer が見つかったか？ (explanationは任意とする場合もあるが、今回は必須とする)
    success = bool(thoughts_match and answer_match and explanation_match)

    if success:
        # 6. 新しい形式で整形済みテキストを作成
        structured_response = f"<think>\n{thoughts}\n</think>\nanswer: {extracted_answer}\nexplanation: {explanation}"
        return structured_response, True, extracted_answer
    else:
        # 失敗の詳細をログに出力するとデバッグに役立つ
        # print(f"Extraction failed: think={bool(thoughts_match)}, answer={bool(answer_match)}, explanation={bool(explanation_match)}")
        # print(f"Content after think: {content_after_think[:200]}") # デバッグ用
        return None, False, None


### --- DPOペア生成関数 --- ###
# --- [修正箇所 6] create_dpo_pair の chosen_response_text 組み立て変更 ---
def create_dpo_pair(example, client, model_name, use_system_prompt_arg, temperature, top_p):
    example_id = example.get('id', 'N/A')
    prompt_data = example.copy()

    # --- Chosen Response の準備 (タグ形式調整) ---
    chosen_cot = example.get('cot', '') # 元のデータは'cot'列にあると仮定
    chosen_expl = example.get('explanation', '')
    chosen_ans_formatted = format_answer(example.get('answer'))

    # chosenレスポンスを新しい形式で組み立てる
    chosen_response_text = f"<think>\n{chosen_cot}\n</think>\nanswer: {chosen_ans_formatted}\nexplanation: {chosen_expl}"

    # 簡単な検証
    if not chosen_cot or not chosen_expl or not chosen_ans_formatted:
         # chosenのデータが不完全な場合はスキップ
         return None, {'error': f'Chosen data incomplete (cot, expl, or answer missing) for {example_id}.'}

    # --- Rejected Response の生成 ---
    rejected_prompt_for_api, target_rejected_choice = create_rejected_prompt(prompt_data)

    if rejected_prompt_for_api is None:
        return None, {'error': f'Could not determine target rejected choice for {example_id}. Skipping.'}

    rejected_raw_response = call_gemini_for_rejected(
        rejected_prompt_for_api, client, model_name,
        system_prompt=None,
        temperature=temperature, top_p=top_p
    )

    # 応答の抽出と検証 (extract_content_for_dpo は新しい形式に対応済み)
    rejected_response_text, rejected_success, rejected_answer_extracted = extract_content_for_dpo(rejected_raw_response)
    generation_info = {'success': False, 'error': '', 'raw_response': rejected_raw_response or "", 'target_rejected': target_rejected_choice}

    if rejected_success and rejected_response_text:
        # a) 生成されたrejectedのanswerが、指示したtarget_rejected_choiceと一致するか？
        if str(rejected_answer_extracted).strip() != str(target_rejected_choice).strip():
            generation_info['error'] = f'Rejected answer "{rejected_answer_extracted}" does not match target "{target_rejected_choice}".'
            print(f"Warning for {example_id}: {generation_info['error']}")
            return None, generation_info

        # b) chosenとrejectedが同一でないか？ (空白無視で比較)
        if ''.join(chosen_response_text.split()) == ''.join(rejected_response_text.split()):
            generation_info['error'] = 'Rejected identical to chosen.'
            return None, generation_info
        # c) rejectedが空でないか？
        if not rejected_response_text.strip():
             generation_info['error'] = 'Rejected response is empty after parsing.'
             return None, generation_info

        generation_info['success'] = True

        # DPO prompt カラムの値の決定 (変更なし)
        dpo_prompt_column_value = create_user_prompt(prompt_data)

        dpo_pair = {
            "prompt": dpo_prompt_column_value,
            "chosen": chosen_response_text,
            "rejected": rejected_response_text,
            "original_id": example_id,
        }
        return dpo_pair, generation_info
    else:
        # 抽出失敗 or API呼び出し失敗
        generation_info['error'] = generation_info.get('error', 'Failed to extract/parse rejected response.')
        if not rejected_raw_response and not generation_info.get('error'):
             generation_info['error'] = 'API call failed or empty.'
        # 抽出失敗の詳細を追加
        if rejected_raw_response and not rejected_success:
             generation_info['error'] += f" | Extraction failed. Raw: {rejected_raw_response[:100]}..."

        return None, generation_info


### --- 引数パーサー定義 --- ###
# (変更なし)
def parse_arguments():
    parser = argparse.ArgumentParser(description="JMLE DPO Dataset Generator using Targeted Mistake Simulation (think tag format)")
    # ... (引数の定義は変更なし) ...
    # 入力データ関連
    parser.add_argument('--sft_dataset_id', type=str, default="doctorin/JMLE-CoT-gemini-2.5-pro-dataset",
                        help='Hugging Face Hub ID of the source SFT dataset.')
    parser.add_argument('--sft_split', type=str, default='train',
                        help='Split of the SFT dataset to use.')
    parser.add_argument('--use_system_prompt_for_sft', action='store_true',
                        help='(Legacy, no direct effect on new rejected generation) Specify if the SFT data was generated using system prompt.')

    # 生成パラメータ
    parser.add_argument('--gemini_model_name', type=str, default="gemini-2.5-pro-exp-03-25",
                        help='Gemini model name for generating rejected responses.')
    parser.add_argument('--gemini_base_url', type=str, default="https://generativelanguage.googleapis.com/v1beta/",
                        help='Base URL for the Gemini API.')
    parser.add_argument('--rejected_temperature', type=float, default=0.6,
                        help='Temperature for generating rejected responses.')
    parser.add_argument('--rejected_top_p', type=float, default=0.9,
                        help='Top-P for generating rejected responses.')

    # 処理制御
    parser.add_argument('--sample_size', type=int, default=0,
                        help='Number of samples to process (0=all). Samples are taken from the beginning.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for shuffling if sample_size > 0 and for rejected choice selection.')
    parser.add_argument('--shuffle_before_select', action='store_true',
                        help='Shuffle the dataset before selecting samples (if sample_size > 0).')


    # 出力と保存
    parser.add_argument('--output_dir', type=str, default="JMLE-DPO-gemini-2.5-pro-dataset",
                        help='Local directory to save the output dataset and logs.')
    parser.add_argument('--save_interim', action='store_true',
                        help='Save interim results during generation.')
    parser.add_argument('--interim_interval', type=int, default=100,
                        help='Interval (number of processed samples) for saving interim results.')

    # Hugging Face Hubへのアップロード
    parser.add_argument('--upload_to_hub', action='store_true',
                        help='Upload the final dataset to Hugging Face Hub.')
    parser.add_argument('--hub_repo_id', type=str, default=None,
                        help='Repository ID on Hugging Face Hub (e.g., your_username/your_repo_name). Required if --upload_to_hub is set.')
    parser.add_argument('--hub_private', action='store_true', default=True,
                        help='Make the repository private on Hugging Face Hub.')

    return parser.parse_args()

### --- メイン処理 --- ###
# (変更なし、中間保存のトリガーが処理数ベースになっている点に注意)
def main():
    args = parse_arguments()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    if args.upload_to_hub and not args.hub_repo_id:
        raise ValueError("--upload_to_hub requires --hub_repo_id to be specified.")
    if args.upload_to_hub and not hf_logged_in:
        print("警告: --upload_to_hubが指定されましたが、Hugging Faceにログインしていません。アップロードはスキップされます。")
        args.upload_to_hub = False

    client = create_client(gemini_api_key, args.gemini_base_url)

    print(f"Loading SFT dataset from Hub: {args.sft_dataset_id}, Split: {args.sft_split}")
    try:
        sft_data = load_dataset(args.sft_dataset_id, split=args.sft_split, token=hf_token if hf_logged_in else None)
        print(f"Loaded {len(sft_data)} samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    total_samples = len(sft_data)
    num_to_process = total_samples
    if args.sample_size > 0 and args.sample_size < total_samples:
        num_to_process = args.sample_size
        print(f"Processing {num_to_process} samples.")
        if args.shuffle_before_select:
            print(f"Shuffling dataset with seed {args.random_seed} before selecting samples.")
            sft_samples = sft_data.shuffle(seed=args.random_seed).select(range(num_to_process))
        else:
            print(f"Selecting the first {num_to_process} samples (no shuffle).")
            sft_samples = sft_data.select(range(num_to_process))
    else:
        print(f"Processing all {total_samples} samples.")
        sft_samples = sft_data

    dpo_pairs = []
    failed_generation_info = []
    print(f"Generating rejected responses using Targeted Mistake Simulation (model={args.gemini_model_name}, temp={args.rejected_temperature}, top_p={args.rejected_top_p})...")
    progress_bar = tqdm(total=len(sft_samples), desc="Generating DPO pairs")
    successful_pairs = 0
    failed_count = 0
    processed_count = 0

    # 中間保存用のリスト (メモリ効率のため、定期的にファイルに書き出してクリアする)
    interim_dpo_pairs_buffer = []
    interim_failed_info_buffer = []

    for i, example in enumerate(sft_samples):
        processed_count += 1
        example_dict = dict(example)
        dpo_pair, gen_info = create_dpo_pair(
            example_dict, client, args.gemini_model_name, args.use_system_prompt_for_sft,
            args.rejected_temperature, args.rejected_top_p
        )
        if dpo_pair:
            dpo_pairs.append(dpo_pair)
            interim_dpo_pairs_buffer.append(dpo_pair) # 中間保存バッファに追加
            successful_pairs += 1
        else:
            gen_info['original_id'] = example_dict.get('id', 'N/A')
            failed_generation_info.append(gen_info)
            interim_failed_info_buffer.append(gen_info) # 中間保存バッファに追加
            failed_count +=1
        progress_bar.update(1)
        progress_bar.set_postfix({"Success": successful_pairs, "Failed": failed_count})

        # --- 中間保存 ---
        if args.save_interim and processed_count > 0 and processed_count % args.interim_interval == 0:
            print(f"\nSaving interim results (processed {processed_count} samples)...")
            os.makedirs(args.output_dir, exist_ok=True)
            # 成功ペアを追記モードでParquetに保存 (スキーマ一貫性のため注意が必要かも)
            # NOTE: Parquetへの追記は複雑な場合があるので、JSONLでの追記や、ファイル分割の方が安全な場合がある
            # ここでは簡単化のため、バッファをDataFrameにして追記する試み (既存ファイルがない場合は新規作成)
            if interim_dpo_pairs_buffer:
                interim_success_df = pd.DataFrame(interim_dpo_pairs_buffer)
                parquet_path = f"{args.output_dir}/dpo_interim_success.parquet"
                if os.path.exists(parquet_path):
                    interim_success_df.to_parquet(parquet_path, index=False, engine='pyarrow', append=True)
                else:
                    interim_success_df.to_parquet(parquet_path, index=False, engine='pyarrow')
                interim_dpo_pairs_buffer = [] # バッファクリア

            # 失敗情報を追記モードでJSONLに保存
            if interim_failed_info_buffer:
                 with open(f"{args.output_dir}/failed_info_interim.jsonl", "a", encoding='utf-8') as f:
                    for info in interim_failed_info_buffer:
                        f.write(json.dumps(info, ensure_ascii=False) + "\n")
                 interim_failed_info_buffer = [] # バッファクリア

            print(f"Interim results saved (Total Success: {successful_pairs}, Total Failed: {failed_count}).")

    progress_bar.close()
    print(f"\nGeneration complete.")
    print(f"Successfully created {successful_pairs} DPO pairs.")
    print(f"Failed or skipped to generate rejected response for {failed_count} samples.")

    # --- 最終データセットの準備と保存 ---
    final_dataset_dict = None
    if not dpo_pairs:
        print("No DPO pairs were generated. Saving only failure log.")
    else:
        # 中間保存で追記していた場合、最後に全データをまとめる必要がある
        # 簡単化のため、ここではメモリ上の dpo_pairs を使う (大規模データでは要見直し)
        dpo_df = pd.DataFrame(dpo_pairs)
        os.makedirs(args.output_dir, exist_ok=True)

        # Parquet形式で保存
        try:
            parquet_path = f"{args.output_dir}/dpo_dataset.parquet"
            dpo_df.to_parquet(parquet_path, index=False)
            print(f"Final DPO dataset saved as Parquet: {parquet_path}")
        except Exception as e:
            print(f"Error saving final dataset as Parquet: {e}")

        # Hugging Face Datasets形式で準備
        try:
            dpo_features = Features({
                'prompt': Value('string'),
                'chosen': Value('string'),
                'rejected': Value('string'),
                'original_id': Value('string'),
            })
            final_dpo_dataset = Dataset.from_pandas(dpo_df, features=dpo_features)
            final_dataset_dict = DatasetDict({"train": final_dpo_dataset})
            save_path = os.path.join(args.output_dir, "hf_dataset")
            final_dataset_dict.save_to_disk(save_path)
            print(f"DPO dataset also saved in HF format locally: {save_path}")
        except Exception as e:
            print(f"Error creating or saving in HF Datasets format: {e}")
            final_dataset_dict = None

    # 失敗情報の最終保存
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        failed_log_path = f"{args.output_dir}/failed_generation_info_total.jsonl"
        # 中間保存を使っていた場合も、最後に全件書き出すのが確実
        with open(failed_log_path, "w", encoding='utf-8') as f:
            for info in failed_generation_info:
                f.write(json.dumps(info, ensure_ascii=False) + "\n")
        print(f"Total failed/skipped generation info saved: {failed_log_path}")
    except Exception as e:
        print(f"Error saving total failed info: {e}")

    # --- Hugging Face Hubへのアップロード ---
    if args.upload_to_hub and final_dataset_dict:
        print(f"\nUploading dataset to Hugging Face Hub: {args.hub_repo_id}")
        try:
            final_dataset_dict.push_to_hub(
                repo_id=args.hub_repo_id,
                private=args.hub_private,
                token=hf_token
            )
            print("Dataset successfully uploaded to Hugging Face Hub.")
            hub_url = f"https://huggingface.co/datasets/{args.hub_repo_id}"
            print(f"Dataset available at: {hub_url}")
        except Exception as e:
            print(f"Error uploading dataset to Hugging Face Hub: {e}")
            print("Please check your repository ID, token permissions, and network connection.")
    elif args.upload_to_hub and not final_dataset_dict:
        print("\nSkipping upload to Hugging Face Hub because the Dataset object could not be created or no pairs were generated.")
    elif not args.upload_to_hub:
         print("\nSkipping upload to Hugging Face Hub as --upload_to_hub was not specified.")


    print("\nScript finished.")

if __name__ == "__main__":
    main()