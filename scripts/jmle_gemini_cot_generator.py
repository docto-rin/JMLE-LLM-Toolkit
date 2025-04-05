# -*- coding: utf-8 -*-
"""JMLE-Gemini-2.5-Pro-CoT-Dataset Generator

この改良版では:
1. 元データセットの問題IDと問題文を保持
2. 生成成功と失敗を明確に分離
3. 後のマッチング不要のデータ構造を提供
4. 各テーブル間で一貫した列構造を確保
5. システムプロンプトとユーザープロンプトの両方をサポート
"""

import os
from dotenv import load_dotenv
from huggingface_hub import login as hf_login, HfApi, HfFolder
from datasets import load_dataset, DatasetDict, Dataset, Features, Value, Sequence
import re
from openai import OpenAI
from py2slack import slack_notify, send_slack
import pandas as pd
from tqdm import tqdm
import time
import random
import argparse
import numpy as np

# 環境変数の読み込み
load_dotenv(override=True)

# 環境変数からトークンを取得
hf_token = os.getenv('HF_TOKEN')
gemini_api_key = os.getenv('GEMINI_API_KEY')

# 必須環境変数のチェック
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY 環境変数が設定されていません。.envファイルか環境変数を確認してください。")

# Hugging Faceへのログイン
if hf_token:
    hf_login(hf_token)
    print("Hugging Faceにログインしました")
else:
    print("HF_TOKEN が設定されていません。HuggingFaceへのアップロードには認証が必要です。")

# トークンの確認
token = HfFolder.get_token()
api = HfApi()
user_info = api.whoami(token=token)
print(user_info)

# OpenAIクライアントの設定
model_name = "gemini-2.5-pro-exp-03-25"
config = {
    "api_key": gemini_api_key,
    "base_url": "https://generativelanguage.googleapis.com/v1beta/",
    "model_name": model_name,
    "client_type": "openai",
    "supports_vision": True,
    "system_role": "system",
    "parameters": {}
}

# 安全にAPIクライアントを作成
def create_client():
    return OpenAI(api_key=config["api_key"], base_url=config["base_url"])

# システムプロンプトの作成関数
def create_system_prompt():
    """システムプロンプトを作成する関数"""
    return (
        "あなたは医師国家試験の過去問を解くにあたり、"
        "医学を隅々まで理解している非常に優秀で論理的なアシスタントです。\n\n"

        "【重要ルール】\n"
        "1. <Thoughts> ～ </Thoughts> の間に、論理的推論プロセスをステップごとに書いてください。\n"
        "   （普段は非公開ですが、今回は出力してください。）\n"
        "2. <o> ～ </o> でくくり、次の形式で必ず出力してください:\n"
        "   answer: [最終的な答え]\n"
        "   explanation: [医学的な根拠や理由]\n\n"
        "3. 問題文に「2つ選べ」などの明確な記載がある場合は、answer に複数の選択肢をカンマ区切り(a, bなど)で書いてください。\n"
        "4. 五者択一の場合はanswerを1つ(a, b, c, d, eのいずれか)にしてください。\n"
        "5. 数値入力問題の場合はanswerを求める数値だけにしてください（小数点以下の四捨五入などは問題文の指示に従う）。\n"
        "6. タグは必ず正しく閉じ、Markdown記法や余計な装飾はしないでください。"
    )

# ユーザープロンプト（システムプロンプト利用時）の作成関数
def create_user_prompt(example: dict) -> str:
    """システムプロンプト使用時のユーザープロンプトを作成する関数"""
    # 選択肢の有無でパターンをざっくり判定
    choice_part = ""
    if example["choices"]:
        choice_part = f"【選択肢】\n{', '.join(example['choices'])}\n\n"
    else:
        choice_part = "【選択肢】\n本問は数値入力問題（選択肢なし）\n\n"

    return (
        "以下の問題文・選択肢・正解ラベルを踏まえ、"
        "最終解答に至るまでの詳細な思考過程と、最終解答を指定のタグで出力してください。\n\n"

        f"【問題番号】{example['id']}\n"
        f"【問題文】\n{example['question']}\n\n"
        f"{choice_part}"
        f"【正解（参考）】\n{example['answer']}\n\n"

        "【出力例】\n"
        "<Thoughts>\n"
        "ここにステップごとの思考過程を詳述する。"
        "問題文のキーワードや病態生理、関連知識などを整理して論理的に推論する。\n"
        "</Thoughts>\n"
        "<o>\n"
        "answer: a\n"
        "explanation: ここに短い医学的根拠や理由を簡潔に書く\n"
        "</o>\n\n"

        "それでは、上記ルールに従って出力してください。"
    )

# 統合プロンプト（従来の方式）の作成関数
def create_unified_prompt(example: dict) -> str:
    """従来のユーザープロンプトとシステムプロンプトを統合した形式"""
    # 選択肢の有無でパターンをざっくり判定
    choice_part = ""
    if example["choices"]:
        choice_part = f"【選択肢】\n{', '.join(example['choices'])}\n\n"
    else:
        choice_part = "【選択肢】\n本問は数値入力問題（選択肢なし）\n\n"

    return (
        "あなたは医師国家試験の過去問を解くにあたり、"
        "医学を隅々まで理解している非常に優秀で論理的なアシスタントです。\n\n"

        "以下の問題文・選択肢・正解ラベルを踏まえ、"
        "最終解答に至るまでの詳細な思考過程と、最終解答を指定のタグで出力してください。\n\n"

        f"【問題番号】{example['id']}\n"
        f"【問題文】\n{example['question']}\n\n"
        f"{choice_part}"
        f"【正解（参考）】\n{example['answer']}\n\n"

        "【重要ルール】\n"
        "1. <Thoughts> ～ </Thoughts> の間に、論理的推論プロセスをステップごとに書いてください。\n"
        "   （普段は非公開ですが、今回は出力してください。）\n"
        "2. <o> ～ </o> でくくり、次の形式で必ず出力してください:\n"
        "   answer: [最終的な答え]\n"
        "   explanation: [医学的な根拠や理由]\n\n"
        "3. 問題文に「2つ選べ」などの明確な記載がある場合は、answer に複数の選択肢をカンマ区切り(a, bなど)で書いてください。\n"
        "4. 五者択一の場合はanswerを1つ(a, b, c, d, eのいずれか)にしてください。\n"
        "5. 数値入力問題の場合はanswerを求める数値だけにしてください（小数点以下の四捨五入などは問題文の指示に従う）。\n"
        "6. タグは必ず正しく閉じ、Markdown記法や余計な装飾はしないでください。\n\n"

        "【出力例】\n"
        "<Thoughts>\n"
        "ここにステップごとの思考過程を詳述する。"
        "問題文のキーワードや病態生理、関連知識などを整理して論理的に推論する。\n"
        "</Thoughts>\n"
        "<o>\n"
        "answer: a\n"
        "explanation: ここに短い医学的根拠や理由を簡潔に書く\n"
        "</o>\n\n"

        "それでは、上記ルールに従って出力してください。"
    )

# Gemini API呼び出し関数（システムプロンプト対応版）
def call_gemini(prompt: str, client=None, retry_count=0, system_prompt=None) -> str:
    if client is None:
        client = create_client()
    
    # システムプロンプトの有無に応じてメッセージを構築
    if system_prompt:
        messages = [
            {"role": config["system_role"], "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    
    try:
        response = client.chat.completions.create(
            model=config["model_name"],
            messages=messages,
            **config["parameters"]
        )
        if response and response.choices:
            return response.choices[0].message.content
        else:
            return None
    except Exception as e:
        # レート制限エラーやネットワークエラーの場合は少し待ってリトライ
        if retry_count < 3:
            wait_time = (2 ** retry_count) * 5  # 指数バックオフ: 5秒, 10秒, 20秒
            print(f"API呼び出しエラー: {str(e)}, {wait_time}秒後にリトライします...")
            time.sleep(wait_time)
            return call_gemini(prompt, client, retry_count + 1, system_prompt)
        else:
            print(f"最大リトライ回数に達しました。エラー: {str(e)}")
            return None

def extract_content(response, original_answer):
    """レスポンスからThoughts、Answer、Explanationを抽出"""
    if response is None:
        return None, None, None, False
    
    thoughts_match = re.search(r"<Thoughts>(.*?)</Thoughts>", response, re.DOTALL)
    output_match = re.search(r"<o>(.*?)</o>", response, re.DOTALL)
    
    if output_match:
        output_content = output_match.group(1)
        answer_match = re.search(r"answer:\s*(.*?)(?:\n|$)", output_content, re.DOTALL)
        explanation_match = re.search(r"explanation:\s*(.*?)(?:\n|$)", output_content, re.DOTALL)
    else:
        answer_match = None
        explanation_match = None
    
    thoughts = thoughts_match.group(1).strip() if thoughts_match else ""
    extracted_answer = answer_match.group(1).strip() if answer_match else ""
    explanation = explanation_match.group(1).strip() if explanation_match else ""
    
    # プロンプト形式に合わせて正解チェックロジックを改善
    is_correct = check_answer_match(extracted_answer, original_answer)
    
    return thoughts, extracted_answer, explanation, is_correct

def check_answer_match(extracted_answer, original_answer):
    """
    回答形式に応じた正解比較を行う
    
    - 単一選択肢: 'a', 'b', 'c', 'd', 'e'
    - 複数選択肢: 'a, b', 'a,b', 'b,a' など順序やスペースの違いを許容
    - 数値回答: 空白や書式の違いを許容
    
    original_answerはリスト形式またはスカラー値の場合あり
    """
    # 空の回答はミスマッチ
    if not extracted_answer:
        return False
        
    # original_answerがリスト形式の場合の処理
    if isinstance(original_answer, list):
        # 数値回答の場合
        if len(original_answer) == 1 and str(original_answer[0]).replace('.', '', 1).isdigit():
            try:
                # 数値比較
                original_num = float(original_answer[0])
                extracted_num = float(extracted_answer.strip())
                return original_num == extracted_num
            except ValueError:
                return False
                
        # 複数選択肢の場合: original_answerのリスト要素を比較
        extracted_choices = [choice.strip().lower() for choice in extracted_answer.split(',')]
        original_choices = [str(choice).strip().lower() for choice in original_answer]
        
        # ソートして順序に依存しない比較
        return sorted(original_choices) == sorted(extracted_choices)
    
    # original_answerが文字列の場合の処理
    else:
        original_str = str(original_answer)
        
        # 数値回答かどうかチェック
        if original_str.replace('.', '', 1).isdigit():
            try:
                original_num = float(original_str)
                extracted_num = float(extracted_answer.strip())
                return original_num == extracted_num
            except ValueError:
                return False
                
        # カンマを含む場合は複数選択肢と判断
        elif ',' in original_str or ',' in extracted_answer:
            original_choices = [choice.strip().lower() for choice in original_str.split(',')]
            extracted_choices = [choice.strip().lower() for choice in extracted_answer.split(',')]
            return sorted(original_choices) == sorted(extracted_choices)
            
        # 単一選択肢の場合
        else:
            return extracted_answer.strip().lower() == original_str.strip().lower()

def process_example(example, client, use_system_prompt=False, progress_bar=None):
    """単一の問題例を処理して結果を返す"""
    example_id = example.get('id', example.get('idx', None))
    
    # 元の回答をそのまま保持
    original_answer = example['answer']
    
    # プロンプト作成（システムプロンプト使用有無に応じて）
    if use_system_prompt:
        system_prompt = create_system_prompt()
        user_prompt = create_user_prompt(example)
    else:
        system_prompt = None
        user_prompt = create_unified_prompt(example)
    
    max_retries = 3
    attempt = 0
    
    # 辞書に元のデータをコピー
    result = {
        'id': example_id,  # 元データのID
        'question': example['question'],
        'choices': example['choices'],
        'answer': example['answer'],
        'cot': "",           # 全ての行に空の値を設定（後で更新）
        'explanation': "",   # 全ての行に空の値を設定（後で更新）
        'generation_info': {  # メタ情報は別の辞書にまとめる
            'success': False,       # 初期値はFalse
            'answer_original': str(original_answer),  # 表示用に文字列化
            'answer_updated': "",
            'attempts': 0,
            'error_info': "",
            'used_system_prompt': use_system_prompt
        }
    }
    
    while attempt < max_retries:
        response = call_gemini(user_prompt, client, system_prompt=system_prompt)
        thoughts, extracted_answer, explanation, is_correct = extract_content(response, original_answer)
        
        # 応答が成功した場合
        if is_correct:
            result.update({
                'cot': thoughts,
                'explanation': explanation,
            })
            result['generation_info'].update({
                'answer_updated': extracted_answer,
                'success': True,
                'attempts': attempt + 1,
                'error_info': ""  # 明示的に空文字列を設定
            })
            if progress_bar:
                progress_bar.update(1)
            return result
        
        # 失敗した場合
        attempt += 1
        if progress_bar:
            progress_bar.set_description(f"Attempt {attempt}/{max_retries}")
    
    # 全リトライ失敗した場合
    result.update({
        'cot': thoughts if thoughts else "",
        'explanation': explanation if explanation else "",
    })
    result['generation_info'].update({
        'answer_updated': extracted_answer if extracted_answer else "",
        'attempts': max_retries,
        'error_info': f"Answer mismatch: expected {original_answer}, got {extracted_answer}"
    })
    
    if progress_bar:
        progress_bar.update(1)
    
    return result

def parse_arguments():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description='JMLE CoT生成ツール')
    
    # データセット関連
    parser.add_argument('--dataset', type=str, default="longisland3/NMLE", 
                        help='使用するデータセットのHugging Face ID')
    parser.add_argument('--sample_size', type=int, default=0,
                        help='処理するサンプル数（0=全て）')
    
    # 出力関連
    parser.add_argument('--output_dir', type=str, default="JMLE-CoT-gemini-2.5-pro-dataset",
                        help='ローカル保存先ディレクトリ')
    parser.add_argument('--save_interim', action='store_true',
                        help='中間結果を保存するかどうか')
    parser.add_argument('--interim_interval', type=int, default=100,
                        help='中間結果の保存間隔（サンプル数）')
    
    # HuggingFaceへのアップロード関連
    parser.add_argument('--upload_to_hub', action='store_true', default=True,
                        help='Hugging Faceにアップロードするかどうか')
    parser.add_argument('--hub_repo', type=str, default="doctorin/JMLE-CoT-gemini-2.5-pro-dataset",
                        help='アップロード先のリポジトリ名（例: username/repo-name）')
    parser.add_argument('--private_repo', action='store_true', default=True,
                        help='プライベートリポジトリにするかどうか')
    
    # Slack通知関連
    parser.add_argument('--notify_slack', action='store_true',
                        help='Slack通知を送信するかどうか')
    
    # プロンプト関連（新規追加）
    parser.add_argument('--use_system_prompt', action='store_true',
                        help='システムプロンプトを使用するかどうか（デフォルトは統合プロンプト）')
    
    return parser.parse_args()

@slack_notify
def main():
    # コマンドライン引数の解析
    args = parse_arguments()
    
    # データセットの読み込み
    print("データセットを読み込み中...")
    dataset = load_dataset(args.dataset)
    
    # データセットの情報を表示
    print(f"データセット情報: {dataset}")
    print(f"サンプル数: {len(dataset['train'])}")
    
    # プロンプト方式の表示
    if args.use_system_prompt:
        print("システムプロンプト + ユーザープロンプト方式を使用します")
    else:
        print("統合プロンプト方式を使用します")
    
    # 処理するサンプル数の設定
    if args.sample_size > 0:
        sample_size = min(args.sample_size, len(dataset['train']))
        print(f"{sample_size}サンプルを処理します（無作為抽出）")
        indices = random.sample(range(len(dataset['train'])), sample_size)
        dataset_samples = [dataset['train'][i] for i in indices]
    else:
        print(f"全{len(dataset['train'])}サンプルを処理します")
        dataset_samples = dataset['train']
    
    # APIクライアントの作成
    client = create_client()
    
    # 結果を格納するリスト
    results = []
    
    # プログレスバー付きで処理
    with tqdm(total=len(dataset_samples), desc="処理中") as progress_bar:
        for i, example in enumerate(dataset_samples):
            # IDフィールドを追加（元データにない場合）
            if 'id' not in example and 'idx' not in example:
                example['id'] = str(i)  # 文字列型に統一
            elif 'id' in example and not isinstance(example['id'], str):
                example['id'] = str(example['id'])  # 既存のIDを文字列型に変換
            
            # サンプルを処理（システムプロンプト使用設定を渡す）
            result = process_example(example, client, args.use_system_prompt, progress_bar)
            results.append(result)
            
            # 定期的に中間結果を保存
            if args.save_interim and (i + 1) % args.interim_interval == 0:
                print(f"\n{i + 1}サンプル処理完了。中間結果を保存しています...")
                interim_df = pd.DataFrame(results)
                os.makedirs(args.output_dir, exist_ok=True)
                interim_df.to_csv(f"{args.output_dir}/interim_{i + 1}.csv", index=False)
    
    # 結果をDataFrameに変換
    results_df = pd.DataFrame(results)
    
    # 成功と失敗の統計
    success_count = sum(result['generation_info']['success'] for result in results)
    failure_count = len(results) - success_count
    success_rate = success_count / len(results) * 100
    
    print(f"\n処理完了!")
    print(f"成功: {success_count} ({success_rate:.2f}%)")
    print(f"失敗: {failure_count} ({100 - success_rate:.2f}%)")
    
    # 成功したサンプルと失敗したサンプルを分離
    successful_samples = [r for r in results if r['generation_info']['success']]
    failed_samples = [r for r in results if not r['generation_info']['success']]
    
    successful_df = pd.DataFrame(successful_samples) if successful_samples else pd.DataFrame(columns=[
        'id', 'question', 'choices', 'answer', 'cot', 'explanation', 'generation_info'
    ])
    
    failed_df = pd.DataFrame(failed_samples) if failed_samples else pd.DataFrame(columns=[
        'id', 'question', 'choices', 'answer', 'cot', 'explanation', 'generation_info'
    ])
    
    # 固定の特徴量を定義 - 両方のデータセットで同じ構造を確保
    features = Features({
        'id': Value('string'),
        'question': Value('string'),
        'choices': Sequence(Value('string')),
        'answer': Sequence(Value('string')),
        'cot': Value('string'),
        'explanation': Value('string'),
        'generation_info': Value('string')  # JSONとして保存
    })
    
    # generation_info列をJSON文字列に変換
    if not successful_df.empty:
        successful_df['generation_info'] = successful_df['generation_info'].apply(lambda x: str(x))
    
    if not failed_df.empty:
        failed_df['generation_info'] = failed_df['generation_info'].apply(lambda x: str(x))
    
    # 最終的なデータセットを作成 - 空の辞書から始める
    dataset_dict = {}
    
    # 成功したサンプルがある場合のみ、trainスプリットを追加
    if len(successful_samples) > 0:
        successful_dataset = Dataset.from_pandas(successful_df, features=features)
        dataset_dict["train"] = successful_dataset
    
    # 失敗データが存在する場合のみunmatchedスプリットを追加
    if len(failed_samples) > 0:
        failed_dataset = Dataset.from_pandas(failed_df, features=features)
        dataset_dict["unmatched"] = failed_dataset
    
    final_dataset = DatasetDict(dataset_dict)
    
    # ローカルに保存
    os.makedirs(args.output_dir, exist_ok=True)
    final_dataset.save_to_disk(args.output_dir)
    print(f"データセットを {args.output_dir} に保存しました")
    
    # DataFrameもCSVとして保存
    successful_df.to_csv(f"{args.output_dir}/successful_samples.csv", index=False)
    failed_df.to_csv(f"{args.output_dir}/failed_samples.csv", index=False)
    print(f"CSVデータも {args.output_dir} に保存しました")
    
    # Hugging Faceにアップロード
    if args.upload_to_hub and args.hub_repo:
        final_dataset.push_to_hub(args.hub_repo, private=args.private_repo)
        print(f"データセットを {args.hub_repo} にアップロードしました")
        
        # プロンプト方式の情報を付け加える
        prompt_type = "system_user_prompt" if args.use_system_prompt else "unified_prompt"
        repo_info = f"{args.hub_repo} ({prompt_type}方式)"
        
        # Slack通知
        if args.notify_slack:
            send_slack("JMLE-CoT-Updated", f"JMLE-CoT-Updated dataset processing completed and uploaded to {repo_info}.")
    
    return final_dataset

if __name__ == "__main__":
    main()